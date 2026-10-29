"""Tests for sample, model and measurement integrity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_backends import cand
from test_validation_workflows import demo_results

from aidisclose import Results
from aidisclose.aggregate import group_panel
from aidisclose.classify.supervised import SupervisedClassifier
from aidisclose.validate.recall import estimate_recall
from aidisclose.validate.sampling import seal_sample_key, verify_sample_key


class TestFeatureIntegrity(unittest.TestCase):
    def test_sample_seal_detects_joint_deletion_and_changed_weight(self):
        df = pd.DataFrame(
            {
                "sentence_id": ["a", "b", "c"],
                "text": ["x", "y", "z"],
                "sampling_weight": [1.0, 2.0, 3.0],
                "dataset_sha256": ["abc"] * 3,
            }
        )
        sealed = seal_sample_key(df)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "key.csv"
            sealed.to_csv(f, index=False)
            loaded = pd.read_csv(f, dtype=str, keep_default_na=False)
            verify_sample_key(loaded)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_sample_key(loaded.iloc[:1])
            changed = loaded.copy()
            changed.loc[0, "sampling_weight"] = "99"
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_sample_key(changed)
            verify_sample_key(loaded.iloc[::-1])

    def test_supervised_model_digest_tracks_feature_weights(self):
        c = [
            cand("We deployed artificial intelligence to route trucks in 2024."),
            cand("We launched artificial intelligence to route ships in 2024."),
            cand("We plan to test artificial intelligence for routing in 2025."),
            cand("We plan to test artificial intelligence for shipping in 2025."),
        ]
        m = SupervisedClassifier.train(c, ["SUBSTANTIVE", "SUBSTANTIVE", "WEAK", "WEAK"])
        old = m.config()["fitted_model_sha256"]
        m.pipeline.named_steps["features"].transformer_weights = {"word": 0.0, "char": 0.0}
        self.assertNotEqual(old, m.config()["fitted_model_sha256"])

    def test_overlapping_results_cannot_be_concatenated(self):
        r = demo_results()
        with self.assertRaisesRegex(ValueError, "duplicate filing"):
            Results.concat([r, r])

    def test_unknown_firms_do_not_inflate_full_prevalence(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a"],
                "keyword_any": [1, 1],
                "claim_any": pd.array([1, pd.NA], dtype="Int64"),
                "substantive_any": pd.array([1, pd.NA], dtype="Int64"),
                "classification_complete": [False, False],
                "keyword_per_10k_words": [1, 1],
                "substantive_per_10k_words": [np.nan, np.nan],
                "substantive_share": [np.nan, np.nan],
            }
        )
        row = group_panel(df, by=["g"]).iloc[0]
        self.assertTrue(pd.isna(row.share_substantive))
        self.assertEqual(row.share_substantive_observed, 1)
        self.assertEqual(row.share_substantive_lower_bound, 0.5)
        self.assertEqual(row.share_substantive_upper_bound, 1)
        df["size"] = [float("inf"), 1]
        with self.assertRaisesRegex(ValueError, "finite"):
            group_panel(df, by=["g"], weight="size")

    def test_singleton_stratum_does_not_produce_spurious_interval(self):
        gold = pd.DataFrame(
            {
                "gold_state": ["NO_CLAIM", "SUBSTANTIVE"],
                "recall_stratum": ["probe", "random"],
                "recall_weight": [1.0, 1.0],
            }
        )
        r = estimate_recall(gold, found_substantive=10, n_boot=50)
        self.assertEqual(r["ci"]["recall_substantive"], [None, None])
        self.assertIn("not estimable", r["interval_status"])

    def test_weighted_fit_metadata(self):
        c = [
            cand("We deployed artificial intelligence to route trucks in 2024."),
            cand("We launched artificial intelligence to route ships in 2024."),
            cand("We plan to test artificial intelligence for routing in 2025."),
            cand("We plan to test artificial intelligence for shipping in 2025."),
        ]
        m = SupervisedClassifier.train(c, ["SUBSTANTIVE", "SUBSTANTIVE", "WEAK", "WEAK"], sample_weight=[1, 1, 3, 3])
        self.assertTrue(m.config()["weighted_fit"])


class TestWeightedTrainCLI(unittest.TestCase):
    def test_weighted_option_reaches_final_model_without_cv(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from aidisclose.cli import main

        def run(*args):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(list(map(str, args)))

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            demo_results().save(d / "res")
            run("sample", "--results", d / "res", "--n", "20", "--out", d / "sample.csv")
            sample = pd.read_csv(d / "sample.csv", keep_default_na=False)
            key = pd.read_csv(d / "sample_key.csv", keep_default_na=False)
            sample = sample.merge(key[["sentence_id", "state"]], on="sentence_id", validate="one_to_one")
            sample["gold_state"] = sample.pop("state")
            sample.to_csv(d / "gold.csv", index=False)
            run(
                "train",
                "--gold",
                d / "gold.csv",
                "--key",
                d / "sample_key.csv",
                "--results",
                d / "res",
                "--weighted",
                "--model-out",
                d / "model.joblib",
            )
            self.assertTrue(SupervisedClassifier.load(d / "model.joblib").config()["weighted_fit"])
