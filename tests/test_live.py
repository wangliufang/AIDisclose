"""Live tests against the real data sources (skipped unless AIDISCLOSE_LIVE=1).

They need network access and, for SEC EDGAR, a declared User-Agent:

    AIDISCLOSE_LIVE=1 AIDISCLOSE_USER_AGENT="Jane Doe jane@uni.edu" python -m pytest tests/test_live.py

The assertions are deliberately loose (sources change over time); they check
that the adapters still understand the current API responses.
"""

import os
import tempfile
import unittest

LIVE = os.environ.get("AIDISCLOSE_LIVE") == "1"
UA = os.environ.get("AIDISCLOSE_USER_AGENT", "")


@unittest.skipUnless(LIVE and UA, "set AIDISCLOSE_LIVE=1 and AIDISCLOSE_USER_AGENT to run live tests")
class LiveEdgarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aidisclose.sources import EdgarSource

        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = EdgarSource(UA, cache_dir=cls.tmp.name, max_rps=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_10k(self):
        filings = self.src.list_filings(tickers=["AAPL"], start_year=2023, end_year=2023)
        self.assertGreaterEqual(len(filings), 1)
        self.assertEqual(filings[0].country, "US")
        doc = self.src.fetch_document(filings[0])
        self.assertIn("10-K:1A", doc.sections)
        self.assertEqual(doc.language, "en")

    def test_20f(self):
        filings = self.src.list_filings(tickers=["SAP"], forms=("20-F",), start_year=2023, end_year=2023)
        self.assertGreaterEqual(len(filings), 1)
        self.assertEqual(filings[0].country, "DE")
        doc = self.src.fetch_document(filings[0])
        self.assertTrue(any(k.startswith("20-F:") for k in doc.sections))

    def test_40f_exhibits(self):
        filings = self.src.list_filings(tickers=["CNI"], forms=("40-F",), start_year=2023, end_year=2023)
        self.assertGreaterEqual(len(filings), 1)
        self.assertEqual(filings[0].country, "CA")
        doc = self.src.fetch_document(filings[0])
        self.assertTrue(any(k.startswith("40-F:") and k != "40-F:main" for k in doc.sections), list(doc.sections))


@unittest.skipUnless(LIVE, "set AIDISCLOSE_LIVE=1 to run live tests")
class LiveXbrlEurostatTests(unittest.TestCase):
    def test_xbrl(self):
        from aidisclose.sources import XbrlFilingsSource

        with tempfile.TemporaryDirectory() as tmp:
            src = XbrlFilingsSource(cache_dir=tmp, user_agent=UA or "aidisclose live test")
            filings = src.list_filings(countries=["DK"], start_year=2023, end_year=2023, max_pages=1)
            self.assertGreaterEqual(len(filings), 1)
            english = [f for f in filings if f.language == "en"] or filings
            doc = src.fetch_document(english[0])
            self.assertGreater(doc.n_words, 1000)

    def test_eurostat(self):
        from aidisclose.validate import eurostat_ai_share, eurostat_ai_share_by_industry

        df = eurostat_ai_share()
        self.assertTrue(((df["eurostat_share"] >= 0) & (df["eurostat_share"] <= 1)).all())
        self.assertIn("DK", set(df["country"]))
        ind = eurostat_ai_share_by_industry(countries=["DK"])
        self.assertIn("J", set(ind["nace_section"]))


if __name__ == "__main__":
    unittest.main()
