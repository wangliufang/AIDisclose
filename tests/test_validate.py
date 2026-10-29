import unittest

import numpy as np
import pandas as pd
from helpers import fixture

from aidisclose.validate import (
    blind,
    bootstrap_ci,
    classification_report,
    cohen_kappa,
    compare_with_external,
    confusion_matrix,
    correlation,
    eurostat_ai_share,
    merge_annotations,
    parse_jsonstat,
    report_to_markdown,
    sample_for_annotation,
)
from aidisclose.validate.external import permutation_pvalue

S, W, N = "SUBSTANTIVE", "WEAK", "NO_CLAIM"


class MetricTests(unittest.TestCase):
    y_true = [S, S, S, W, W, N, N, N, N, W]
    y_pred = [S, W, S, W, N, N, N, S, N, W]

    def test_confusion(self):
        m = confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(m.sum(), 10)
        self.assertEqual(m[0, 0], 2)  # S correctly predicted twice

    def test_against_sklearn(self):
        try:
            from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
        except ImportError:  # pragma: no cover
            self.skipTest("scikit-learn not installed")
        rep = classification_report(self.y_true, self.y_pred)
        p, r, f, _ = precision_recall_fscore_support(self.y_true, self.y_pred, labels=[S, W, N], zero_division=0)
        for i, lab in enumerate([S, W, N]):
            self.assertAlmostEqual(rep["per_class"][lab]["precision"], p[i])
            self.assertAlmostEqual(rep["per_class"][lab]["recall"], r[i])
            self.assertAlmostEqual(rep["per_class"][lab]["f1"], f[i])
        self.assertAlmostEqual(rep["macro_f1"], f1_score(self.y_true, self.y_pred, average="macro"))
        self.assertAlmostEqual(cohen_kappa(self.y_true, self.y_pred), cohen_kappa_score(self.y_true, self.y_pred))
        w = [1, 2, 1, 1, 3, 1, 1, 2, 1, 1]
        wrep = classification_report(self.y_true, self.y_pred, weights=w)
        pw, rw, fw, _ = precision_recall_fscore_support(self.y_true, self.y_pred, labels=[S, W, N], sample_weight=w)
        self.assertAlmostEqual(wrep["per_class"][S]["precision"], pw[0])
        self.assertAlmostEqual(wrep["per_class"][W]["recall"], rw[1])

    def test_binary_views(self):
        rep = classification_report(self.y_true, self.y_pred)
        sub = rep["substantive_vs_rest"]
        self.assertAlmostEqual(sub["precision"], 2 / 3)
        self.assertAlmostEqual(sub["recall"], 2 / 3)

    def test_kappa_edge_cases(self):
        self.assertTrue(np.isnan(cohen_kappa([S, S], [S, S])))
        self.assertTrue(np.isnan(cohen_kappa([], [])))

    def test_bootstrap(self):
        pt, lo, hi = bootstrap_ci(
            self.y_true, self.y_pred, lambda t, p: classification_report(t, p)["accuracy"], n_boot=200
        )
        self.assertAlmostEqual(pt, 0.7)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_markdown(self):
        md = report_to_markdown(classification_report(self.y_true, self.y_pred))
        self.assertIn("| SUBSTANTIVE |", md)


class SamplingTests(unittest.TestCase):
    def frame(self):
        rows = []
        for i in range(300):
            state = S if i < 15 else (W if i < 120 else N)
            rows.append(
                {
                    "sentence_id": f"s{i}",
                    "text": f"t{i}",
                    "state": state,
                    "reason": "VAGUE",
                    "raw_terms": "ai_acronym",
                    "firm_name": "F",
                    "fiscal_year": 2023,
                    "section": "full",
                    "prev_text": "",
                    "next_text": "",
                    "confidence": 0.5,
                }
            )
        return pd.DataFrame(rows)

    def test_floor_and_weights(self):
        df = self.frame()
        s = sample_for_annotation(df, n=60, min_per_stratum=15, seed=1)
        counts = s["state"].value_counts()
        self.assertEqual(counts[S], 15)  # rare stratum gets its floor (all 15 available)
        self.assertLessEqual(len(s), 60)
        # weights reproduce the population size of every stratum
        for state, n_pop in df["state"].value_counts().items():
            self.assertAlmostEqual(s.loc[s["state"] == state, "sampling_weight"].sum(), n_pop)
        again = sample_for_annotation(df, n=60, min_per_stratum=15, seed=1)
        self.assertEqual(list(s["sentence_id"]), list(again["sentence_id"]))

    def test_blind_and_merge(self):
        s = sample_for_annotation(self.frame(), n=30, min_per_stratum=5)
        visible, key = blind(s)
        self.assertNotIn("state", visible.columns)
        self.assertNotIn("sampling_weight", visible.columns)
        self.assertIn("state", key.columns)
        visible = visible.copy()
        visible["gold_state"] = key["state"].str.lower()
        merged = merge_annotations(visible, key)
        self.assertEqual(len(merged), len(s))
        self.assertTrue((merged["gold_state"] == merged["state"]).all())


class ExternalTests(unittest.TestCase):
    def test_parse_jsonstat(self):
        df = parse_jsonstat(fixture("eurostat_isoc_eb_ai_synthetic.json").read_bytes())
        self.assertEqual(len(df), 2 * 5 * 2)
        cell = df[(df["size_emp"] == "GE250") & (df["geo"] == "NL") & (df["time"] == "2024")]
        self.assertEqual(cell["value"].item(), 45.0)
        self.assertIn("size_emp_label", df.columns)

    def test_eurostat_share(self):
        ext = eurostat_ai_share(path=fixture("eurostat_isoc_eb_ai_synthetic.json"))
        self.assertEqual(sorted(ext["country"].unique()), ["DK", "GR", "NL", "RO"])  # EL -> GR, EU aggregate dropped
        dk = ext[(ext["country"] == "DK") & (ext["year"] == 2024)]["eurostat_share"].item()
        self.assertAlmostEqual(dk, 0.50)
        small = eurostat_ai_share(path=fixture("eurostat_isoc_eb_ai_synthetic.json"), size_emp="GE10")
        self.assertAlmostEqual(
            small[(small["country"] == "DK") & (small["year"] == 2024)]["eurostat_share"].item(), 0.25
        )

    def test_correlation(self):
        self.assertAlmostEqual(correlation([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
        self.assertAlmostEqual(correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(np.isnan(correlation([1, 1, 1], [1, 2, 3])))
        self.assertAlmostEqual(permutation_pvalue([1, 2, 3, 4], [1, 2, 3, 4]), 2 / 24)

    def test_compare(self):
        ext = eurostat_ai_share(path=fixture("eurostat_isoc_eb_ai_synthetic.json"))
        grp = pd.DataFrame(
            {
                "country": ["DK", "NL", "RO", "GR"],
                "fiscal_year": [2023] * 4,
                "n_firms": [10, 10, 10, 2],
                "share_keyword": [0.9, 0.9, 0.8, 0.9],
                "share_substantive": [0.40, 0.30, 0.05, 0.1],
                "share_claim": [0.6, 0.5, 0.2, 0.3],
            }
        )
        merged, summary = compare_with_external(grp, ext, min_firms=5)
        self.assertEqual(len(merged), 3)  # GR dropped by min_firms
        sub = summary[summary["measure"] == "share_substantive"].iloc[0]
        self.assertAlmostEqual(sub["spearman"], 1.0)
        lagged, _ = compare_with_external(grp, ext, year_lag=1, min_firms=5)
        self.assertTrue((lagged["year"] == 2024).all())


if __name__ == "__main__":
    unittest.main()
