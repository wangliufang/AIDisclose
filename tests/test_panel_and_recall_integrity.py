"""Tests for panel, sampling and recall integrity."""

import unittest

import pandas as pd

from aidisclose.aggregate import _add_dynamics
from aidisclose.validate.recall import estimate_recall, sample_recall
from aidisclose.validate.sampling import sample_for_annotation


class TestFeatureIntegrity(unittest.TestCase):
    def test_dynamics_all_filings_and_empty_hits(self):
        panel = pd.DataFrame(
            {
                "firm_id": ["a", "a", "b", "b"],
                "fiscal_year": [2022, 2023, 2022, 2023],
                "filing_id": ["a22", "a23", "b22", "b23"],
                "substantive_any": [0, 1, 0, 1],
                "n_substantive": [0, 1, 0, 1],
                "classification_complete": [True] * 4,
            }
        )
        s = pd.DataFrame(
            {
                "filing_id": ["a23", "b23"],
                "raw_terms": ["AI", "AI"],
                "text": ["We use AI for logistics", "We use AI for forecasting"],
                "state": ["SUBSTANTIVE"] * 2,
                "classification_status": ["ok"] * 2,
            }
        )
        _add_dynamics(panel, s, 0.8)
        self.assertEqual(panel["n_substantive_new"].tolist()[1::2], [1, 1])
        s["raw_terms"] = ""
        _add_dynamics(panel, s, 0.8)

    def test_allocation_budget_and_coverage(self):
        df = pd.DataFrame({"state": ["a"] * 20 + ["b"] * 20 + ["c"] * 20, "raw_terms": ["AI"] * 60})
        sample = sample_for_annotation(df, n=6, min_per_stratum=20)
        self.assertEqual(len(sample), 6)
        self.assertEqual(set(sample.state), {"a", "b", "c"})
        with self.assertRaises(ValueError):
            sample_for_annotation(df, n=2)

    def test_recall_rejects_invalid_weights_and_nonresponse(self):
        pool = pd.DataFrame({"pool": ["probe", "random"], "inclusion_prob": [0.1, 0.0], "sentence_id": ["a", "b"]})
        with self.assertRaises(ValueError):
            sample_recall(pool, n=2)
        gold = pd.DataFrame(
            {"gold_state": ["SUBSTANTIVE", ""], "recall_weight": [2, 2], "recall_stratum": ["probe", "random"]}
        )
        with self.assertRaises(ValueError):
            estimate_recall(gold, found_substantive=1, n_boot=2)
        gold.gold_state = ["SUBSTANTIVE", "NO_CLAIM"]
        gold.loc[1, "recall_weight"] = float("nan")
        with self.assertRaises(ValueError):
            estimate_recall(gold, found_substantive=1, n_boot=2)
