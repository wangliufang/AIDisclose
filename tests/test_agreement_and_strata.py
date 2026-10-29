"""Regressions for agreement, sampling uncertainty and benchmark coverage."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from test_validation_workflows import demo_results

from aidisclose.cli import main
from aidisclose.validate.annotation import agreement_report
from aidisclose.validate.external import eurostat_ai_share_by_industry
from aidisclose.validate.metrics import bootstrap_report
from aidisclose.validate.sampling import sample_for_annotation


class TestFeatureIntegrity(unittest.TestCase):
    def test_agreement_rejects_ambiguous_or_invalid_ratings(self):
        a = pd.DataFrame({"sentence_id": ["s1", "s2"], "gold_state": ["WEAK", None]})
        b = pd.DataFrame({"sentence_id": ["s1", "s2"], "gold_state": ["WEAK", "WEAK"]})
        self.assertEqual(agreement_report([a, b])["n_items_all_annotators"], 1)
        dup = pd.DataFrame({"sentence_id": ["s1", "s1"], "gold_state": ["NO_CLAIM", "WEAK"]})
        with self.assertRaisesRegex(ValueError, "duplicate ratings"):
            agreement_report([dup, b])
        a.loc[1, "gold_state"] = "nan"
        with self.assertRaisesRegex(ValueError, "invalid gold_state"):
            agreement_report([a, b])
        with self.assertRaisesRegex(ValueError, "annotator identifier"):
            agreement_report(
                pd.DataFrame({"sentence_id": ["s1", "s1"], "annotator": ["a", ""], "gold_state": ["WEAK", "WEAK"]})
            )

    def test_singleton_stratum_has_no_estimated_interval(self):
        report = bootstrap_report(
            ["WEAK", "NO_CLAIM", "NO_CLAIM"], ["WEAK", "NO_CLAIM", "WEAK"], strata=["a", "b", "b"], n_boot=50
        )
        self.assertAlmostEqual(report["accuracy"][0], 2 / 3)
        self.assertTrue(np.isnan(report["accuracy"][1]))
        self.assertTrue(np.isnan(report["accuracy"][2]))
        with self.assertRaisesRegex(ValueError, "strata cannot contain missing"):
            bootstrap_report(["WEAK", "WEAK"], ["WEAK", "WEAK"], strata=["a", None], n_boot=2)

    def test_distinct_strata_cannot_collide_on_delimiters(self):
        df = pd.DataFrame(
            {
                "sentence_id": ["1", "2"],
                "raw_terms": ["x", "x"],
                "one": ["a|b", "a"],
                "two": ["c", "b|c"],
                "state": ["WEAK", "WEAK"],
                "reason": ["VAGUE", "VAGUE"],
            }
        )
        sample = sample_for_annotation(df, n=2, strata=["one", "two"], min_per_stratum=0)
        self.assertEqual(sample.sampling_stratum.nunique(), 2)

    def test_eurostat_local_size_and_country_filters(self):
        obj = {
            "id": ["size_emp", "nace_r2", "geo", "time"],
            "size": [2, 1, 2, 1],
            "dimension": {
                "size_emp": {"category": {"index": {"GE10": 0, "GE250": 1}}},
                "nace_r2": {"category": {"index": {"C": 0}}},
                "geo": {"category": {"index": {"DK": 0, "EL": 1}}},
                "time": {"category": {"index": {"2023": 0}}},
            },
            "value": {"0": 10, "1": 20, "2": 70, "3": 80},
        }
        obj["id"] = ["indic_is", "unit"] + obj["id"]
        obj["size"] = [1, 1] + obj["size"]
        obj["dimension"].update(
            {"indic_is": {"category": {"index": {"E_AI_TANY": 0}}}, "unit": {"category": {"index": {"PC_ENT": 0}}}}
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            path.write_text(json.dumps(obj))
            out = eurostat_ai_share_by_industry(path=path, countries=["GR"])
        self.assertEqual(out.country.tolist(), ["GR"])
        self.assertAlmostEqual(out.eurostat_share.iloc[0], 0.2)

    def test_weighted_evaluation_rejects_incomplete_classification(self):
        def run(*args):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return main(list(map(str, args)))

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            res = demo_results()
            i = res.sentences.index[res.sentences.raw_terms.astype(str).ne("")][0]
            res.sentences.loc[i, "classification_status"] = "error"
            res.sentences.loc[i, ["state", "reason"]] = ""
            res.save(d / "result")
            run("sample", "--results", d / "result", "--n", "20", "--out", d / "sample.csv")
            sample = pd.read_csv(d / "sample.csv", dtype=str, keep_default_na=False)
            key = pd.read_csv(d / "sample_key.csv", dtype=str, keep_default_na=False)
            sample["gold_state"] = "NO_CLAIM"
            sample.to_csv(d / "gold.csv", index=False)
            self.assertTrue(key.classification_complete.eq("false").all())
            with self.assertRaisesRegex(SystemExit, "requires complete classification"):
                run(
                    "evaluate",
                    "--gold",
                    d / "gold.csv",
                    "--results",
                    d / "result",
                    "--key",
                    d / "sample_key.csv",
                    "--weighted",
                    "--cluster-by",
                    "none",
                    "--n-boot",
                    "3",
                )
