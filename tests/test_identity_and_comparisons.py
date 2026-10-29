"""Tests for identity preservation and one-to-one comparisons."""

import tempfile
import unittest

import pandas as pd

from aidisclose import Pipeline, Results, firm_year_panel
from aidisclose.models import Document, Filing
from aidisclose.validate.external import compare_with_external, load_benchmark


def make_document():
    f = Filing(
        source="local",
        filing_id="0001",
        firm_id="000123",
        firm_name="Example",
        fiscal_year=2024,
        country="US",
        industry_code="0123",
        industry_scheme="SIC",
    )
    return Document(
        filing=f,
        language="en",
        text=(
            "We use machine learning to predict failures at hundreds of customer sites. "
            "Our business provides services to customers across the United States."
        ),
    )


class TestFeatureIntegrity(unittest.TestCase):
    def test_benchmark_percent_sign_and_invalid_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/benchmark.csv"
            pd.DataFrame({"country": ["US", "GB"], "year": [2024, 2024], "adoption": ["0.5%", "0.9%"]}).to_csv(
                path, index=False
            )
            out = load_benchmark(path, value="adoption")
            self.assertAlmostEqual(out.benchmark_share.iloc[0], 0.005)
            self.assertAlmostEqual(out.benchmark_share.iloc[1], 0.009)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_benchmark(path, value="adoption", percent=False)
            pd.DataFrame({"country": ["US"], "year": [2024], "adoption": ["125%"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_benchmark(path, value="adoption")

    def test_result_round_trip_preserves_codes_and_joins(self):
        r = Pipeline(recall_random=10).run_documents([make_document()])
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = Results.load(tmp)
        for frame in (loaded.documents, loaded.sentences, loaded.recall_pool):
            if len(frame):
                self.assertEqual(set(frame.filing_id), {"0001"})
                self.assertEqual(set(frame.firm_id), {"000123"})
        self.assertEqual(loaded.documents.industry_code.iloc[0], "0123")
        attrs = pd.DataFrame({"firm_id": ["000123"], "employees": [42]})
        p = firm_year_panel(loaded, firm_attributes=attrs)
        self.assertEqual(p.employees.iloc[0], 42)
        self.assertEqual(loaded.documents.n_substantive.iloc[0], r.documents.n_substantive.iloc[0])

    def test_duplicate_documents_rejected_in_both_entry_points(self):
        d = make_document()
        with self.assertRaisesRegex(ValueError, "duplicate filing"):
            Pipeline().run_documents([d, d])
        with self.assertRaisesRegex(ValueError, "duplicate filing"):
            Pipeline().run([d.filing, d.filing], source=None)

    def test_comparison_rejects_duplicate_and_missing_keys(self):
        g = pd.DataFrame({"country": ["US", "GB", "DE"], "fiscal_year": [2024] * 3, "share_claim": [0.1, 0.2, 0.3]})
        e = pd.DataFrame({"country": ["US", "GB", "DE"], "year": [2024] * 3, "eurostat_share": [0.1, 0.2, 0.3]})
        merged, summary = compare_with_external(g, e, measures=["share_claim"])
        self.assertEqual(len(merged), 3)
        self.assertEqual(summary.n_pairs.iloc[0], 3)
        with self.assertRaisesRegex(ValueError, "external comparison keys must be unique"):
            compare_with_external(g, pd.concat([e, e.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "group comparison keys must be unique"):
            compare_with_external(pd.concat([g, g.iloc[[0]]]), e)
        e.loc[0, "year"] = None
        with self.assertRaisesRegex(ValueError, "missing values"):
            compare_with_external(g, e)
        e.loc[0, "year"] = 2024
        e.loc[0, "country"] = "  "
        with self.assertRaisesRegex(ValueError, "missing values"):
            compare_with_external(g, e)

    def test_year_key_can_precede_industry_key(self):
        g = pd.DataFrame(
            {
                "country": ["US", "GB", "DE"],
                "fiscal_year": [2023] * 3,
                "industry": ["C", "C", "C"],
                "share_claim": [0.1, 0.2, 0.3],
            }
        )
        e = pd.DataFrame(
            {
                "country": ["US", "GB", "DE"],
                "year": [2024] * 3,
                "industry": ["C", "C", "C"],
                "eurostat_share": [0.1, 0.2, 0.3],
            }
        )
        merged, summary = compare_with_external(
            g,
            e,
            measures=["share_claim"],
            year_lag=1,
            group_keys=("country", "fiscal_year", "industry"),
            external_keys=("country", "year", "industry"),
        )
        self.assertEqual(len(merged), 3)
        self.assertEqual(summary.n_pairs.iloc[0], 3)
