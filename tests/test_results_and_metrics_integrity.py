"""Tests for output, sampling and model integrity."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from test_backends import FakeChat, cand

from aidisclose import Document, Filing, Pipeline, Results
from aidisclose.classify.llm import LLMClassifier
from aidisclose.cli import _weights_and_strata
from aidisclose.validate.external import permutation_pvalue
from aidisclose.validate.metrics import classification_report
from aidisclose.validate.recall import merge_recall_key, validate_recall_coverage


def one_result():
    filing = Filing(source="x", filing_id="f", firm_id="firm", firm_name="Firm", fiscal_year=2024)
    doc = Document(
        filing=filing,
        language="en",
        text="Our company has staff in many places around the world. We deployed artificial intelligence systems for logistics in 2023.",
    )
    return Pipeline().run_documents([doc])


class TestFeatureIntegrity(unittest.TestCase):
    def test_invalid_model_response_never_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            chat = FakeChat(["not JSON", '{"state":"SUBSTANTIVE","reason":"DEPLOYMENT"}'])
            clf = LLMClassifier(model="m", session=chat, cache_path=Path(d) / "cache.sqlite", max_retries=0)
            c = cand("We deployed artificial intelligence in logistics.")
            self.assertIn("error", clf.classify_one(c).details)
            self.assertNotIn("error", clf.classify_one(c).details)
            self.assertEqual(len(chat.bodies), 2)
            self.assertNotIn("error", clf.classify_one(c).details)
            self.assertEqual(len(chat.bodies), 2)

    def test_invalid_legacy_cache_is_evicted(self):
        with tempfile.TemporaryDirectory() as d:
            chat = FakeChat(['{"state":"WEAK","reason":"VAGUE"}'])
            clf = LLMClassifier(model="m", session=chat, cache_path=Path(d) / "c.sqlite", max_retries=0)
            c = cand("We deployed artificial intelligence in logistics.")
            messages, _ = clf._messages(c)
            clf.cache.put(clf._key(messages, 0), "invalid", "m", "legacy")
            self.assertNotIn("error", clf.classify_one(c).details)
            self.assertEqual(len(chat.bodies), 1)

    def test_relabel_rejects_short_return(self):
        res = one_result()

        class Bad:
            def classify(self, _):
                return []

            def config(self):
                return {"backend": "bad"}

        p = Pipeline()
        p.classifier = Bad()
        with self.assertRaisesRegex(ValueError, "predictions"):
            p.relabel(res)

    def test_missing_duplicate_and_changed_samples_fail(self):
        key = pd.DataFrame(
            {
                "sentence_id": ["a", "b"],
                "text": ["x", "y"],
                "recall_weight": [2, 3],
                "recall_stratum": ["probe", "random"],
            }
        )
        full = pd.DataFrame({"sentence_id": ["a", "b"], "text": ["x", "y"], "gold_state": ["NO_CLAIM", "WEAK"]})
        self.assertEqual(len(merge_recall_key(full, key)), 2)
        for bad in (full.iloc[:1], pd.concat([full, full.iloc[:1]]), full.assign(text=["x", "changed"])):
            with self.assertRaises(ValueError):
                merge_recall_key(bad, key)
        with self.assertRaises(ValueError):
            merge_recall_key(full, pd.concat([key, key.iloc[:1]]))

    def test_save_empty_pool_and_integrity_check(self):
        res = one_result()
        pool = pd.DataFrame(
            [
                {
                    "sentence_id": "old",
                    "filing_id": "f",
                    "firm_id": "firm",
                    "firm_name": "Firm",
                    "fiscal_year": 2024,
                    "country": "US",
                    "section": "full",
                    "text": "old",
                    "prev_text": "",
                    "next_text": "",
                    "pool": "probe",
                    "probe_terms": "x",
                    "stratum_size": 2,
                    "n_drawn": 1,
                    "inclusion_prob": 0.5,
                }
            ]
        )
        res.recall_pool = pool
        with tempfile.TemporaryDirectory() as d:
            res.save(d)
            self.assertEqual(len(Results.load(d).recall_pool), 1)
            res.recall_pool = pool.iloc[:0]
            res.save(d)
            self.assertEqual(len(Results.load(d).recall_pool), 0)
            with open(Path(d) / "sentences.csv", "a") as fp:
                fp.write("tamper")
            with self.assertRaisesRegex(ValueError, "modified"):
                Results.load(d)

    def test_counts_stable_after_roundtrip(self):
        res = one_result()
        res.sentences.loc[0, ["raw_terms", "state", "reason", "classification_status"]] = [
            "",
            "NO_CLAIM",
            "FALSE_POSITIVE",
            "ok",
        ]
        res.documents.loc[0, ["n_substantive", "n_no_claim", "n_false_positive"]] = [0, 0, 0]
        with tempfile.TemporaryDirectory() as d:
            res.save(d)
            loaded = Results.load(d)
            self.assertEqual(int(loaded.documents.loc[0, "n_no_claim"]), 0)

    def test_macro_f1_and_invalid_inputs(self):
        report = classification_report(["SUBSTANTIVE", "WEAK"], ["SUBSTANTIVE", "SUBSTANTIVE"])
        self.assertAlmostEqual(report["macro_f1"], 1 / 3)
        self.assertEqual(report["per_class"]["WEAK"]["f1"], 0)
        for yt, yp, w in [
            (["WEAK"], [], None),
            (["WEAK"], ["TYPO"], None),
            (["WEAK"], ["WEAK"], [float("nan")]),
            (["WEAK"], ["WEAK"], [-1]),
        ]:
            with self.assertRaises(ValueError):
                classification_report(yt, yp, weights=w)

    def test_block_permutation_order_invariant(self):
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 5, 6, 3, 4]
        b = ["A", "A", "B", "B", "C", "C"]
        order = [0, 2, 4, 1, 3, 5]
        self.assertAlmostEqual(
            permutation_pvalue(x, y, blocks=b),
            permutation_pvalue([x[i] for i in order], [y[i] for i in order], blocks=[b[i] for i in order]),
        )

    def test_weighted_validation_refuses_bad_weights(self):
        for value in (None, float("nan"), -1, 0, float("inf")):
            with self.assertRaises(SystemExit):
                _weights_and_strata(pd.DataFrame({"sampling_weight": [value]}), True)

    def test_missing_first_stage_coverage_refused(self):
        res = one_result()
        res.provenance["recall_pool"] = {"probe_cap_per_document": 0, "random_per_document": 1}
        res.documents.loc[0, "n_probe_sentences"] = 1
        res.documents.loc[0, "n_other_sentences"] = 0
        with self.assertRaisesRegex(ValueError, "zero inclusion"):
            validate_recall_coverage(res)

    def test_partial_save_marker_blocks_load(self):
        res = one_result()
        with tempfile.TemporaryDirectory() as d:
            res.save(d)
            (Path(d) / ".save_in_progress").write_text("interrupted")
            with self.assertRaisesRegex(ValueError, "interrupted"):
                Results.load(d)
