import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeSession

from aidisclose.models import Filing
from aidisclose.sources import EdgarSource, LocalSource, XbrlFilingsSource, fiscal_year_from_period_end, get_source
from aidisclose.sources.base import HttpClient
from aidisclose.sources.xbrl_filings import language_hint

UA = "Test Researcher test@example.org"


def edgar_session():
    return FakeSession(
        {
            "https://data.sec.gov/submissions/CIK0009999901.json": "edgar_submissions.json",
            "https://data.sec.gov/submissions/CIK0009999901-submissions-001.json": "edgar_submissions_older.json",
            "https://www.sec.gov/files/company_tickers.json": "company_tickers.json",
            "https://www.sec.gov/Archives/edgar/data/9999901/": "sample_10k.htm",
        }
    )


class FiscalYearTests(unittest.TestCase):
    def test_convention(self):
        self.assertEqual(fiscal_year_from_period_end("2023-12-31"), 2023)
        self.assertEqual(fiscal_year_from_period_end("2024-01-31"), 2023)
        self.assertEqual(fiscal_year_from_period_end("2024-06-30"), 2024)
        self.assertIsNone(fiscal_year_from_period_end(""))
        self.assertIsNone(fiscal_year_from_period_end("bad"))


class EdgarTests(unittest.TestCase):
    def test_user_agent_required(self):
        with self.assertRaises(ValueError):
            EdgarSource("anonymous", cache_dir=None)
        with self.assertRaises(ValueError):
            EdgarSource(UA, cache_dir=None, max_rps=20)

    def test_list_filings(self):
        sess = edgar_session()
        src = EdgarSource(UA, cache_dir=None, session=sess, max_rps=10)
        filings = src.list_filings(ciks=[9999901], start_year=2021, end_year=2024)
        years = [f.fiscal_year for f in filings]
        self.assertEqual(years, [2021, 2022, 2023, 2024])  # 10-Q and 10-K/A excluded
        f = filings[-1]
        self.assertEqual(f.form, "10-K")
        self.assertEqual(f.country, "US")
        self.assertEqual(f.industry_code, "4213")
        self.assertEqual(f.extra["nace_section"], "H")
        self.assertEqual(f.url, "https://www.sec.gov/Archives/edgar/data/9999901/000999990125000004/nwl-20241231.htm")
        # older submission files are not requested when they predate start_year
        self.assertFalse(any("submissions-001" in u for u, _ in sess.calls))
        # the declared User-Agent is what the SEC requires
        self.assertEqual(src.http.user_agent, UA)

    def test_older_files_and_amendments(self):
        sess = edgar_session()
        src = EdgarSource(UA, cache_dir=None, session=sess)
        filings = src.list_filings(tickers=["NWLX"], start_year=2018, include_amendments=True)
        self.assertIn(2018, [f.fiscal_year for f in filings])
        self.assertIn("10-K/A", [f.form for f in filings])
        self.assertTrue(any("submissions-001" in u for u, _ in sess.calls))
        with self.assertRaises(KeyError):
            src.lookup_ciks(["ZZZZ"])

    def test_fetch_document_sections(self):
        src = EdgarSource(UA, cache_dir=None, session=edgar_session())
        f = src.list_filings(ciks=[9999901], start_year=2024)[0]
        doc = src.fetch_document(f)
        self.assertIn("10-K:1A", doc.sections)
        self.assertEqual(doc.language, "en")


class HttpClientTests(unittest.TestCase):
    def test_cache_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = FakeSession({"https://x.org/a": b"payload"})
            http = HttpClient(UA, max_rps=1000, cache_dir=tmp, session=sess)
            self.assertEqual(http.get_bytes("https://x.org/a"), b"payload")
            self.assertEqual(http.get_bytes("https://x.org/a"), b"payload")
            self.assertEqual(len(sess.calls), 1)  # second call served from the cache

    def test_client_error_not_retried(self):
        import requests

        sess = FakeSession({"https://x.org/missing": (404, b"no")})
        http = HttpClient(UA, max_rps=1000, session=sess, max_retries=3)
        with self.assertRaises(requests.HTTPError):
            http.get_bytes("https://x.org/missing")
        self.assertEqual(len(sess.calls), 1)


class XbrlTests(unittest.TestCase):
    def session(self):
        return FakeSession(
            {
                "https://filings.xbrl.org/api/filings?page%5Bnumber%5D=2": "xbrl_filings_page2.json",
                "https://filings.xbrl.org/api/filings": "xbrl_filings_page1.json",
                "https://filings.xbrl.org/TESTLEI000000000001/": "esef_report.xhtml",
            }
        )

    def test_language_hint(self):
        self.assertEqual(language_hint("/x/reports/vantage-2023-12-31-en.xhtml"), "en")
        self.assertEqual(language_hint("/x/reports/fjord-2023-12-31-da.xhtml"), "da")
        self.assertIsNone(language_hint("/x/reports/kestrel_annual_report_2024.xhtml"))

    def test_build_filter(self):
        flt = XbrlFilingsSource.build_filter(["dk", "se"], 2022, 2023)
        self.assertEqual(flt[0], {"name": "country", "op": "in", "val": ["DK", "SE"]})
        self.assertEqual(flt[1]["val"], "2022-06-01")
        self.assertEqual(flt[2]["val"], "2024-05-31")

    def test_list_filings_paginates_filters_and_dedupes(self):
        sess = self.session()
        src = XbrlFilingsSource(cache_dir=None, session=sess)
        filings = src.list_filings(countries=["DK", "GB"], start_year=2023, end_year=2023)
        names = sorted(f.firm_name for f in filings)
        # Danish-only report of Fjord is dropped; Vantage keeps the English version; Kestrel (no hint) kept
        self.assertEqual(names, ["Kestrel Software plc", "Vantage Retail Group A/S"])
        vantage = next(f for f in filings if f.firm_name.startswith("Vantage"))
        self.assertTrue(vantage.url.endswith("vantage-2023-12-31-en.xhtml"))
        self.assertEqual(vantage.firm_id, "lei:TESTLEI000000000001")
        self.assertEqual(vantage.form, "ESEF")
        kestrel = next(f for f in filings if f.firm_name.startswith("Kestrel"))
        self.assertEqual(kestrel.form, "UKSEF")
        self.assertEqual(kestrel.fiscal_year, 2023)  # period ending March 2024
        first_params = sess.calls[0][1]
        self.assertEqual(first_params["include"], "entity")
        self.assertEqual(json.loads(first_params["filter"])[0]["val"], ["DK", "GB"])
        self.assertEqual(len(sess.calls), 2)

    def test_fetch(self):
        src = XbrlFilingsSource(cache_dir=None, session=self.session())
        f = src.list_filings(countries=["DK"], start_year=2023, end_year=2023)[0]
        doc = src.fetch_document(f)
        self.assertEqual(doc.sections, {})
        self.assertIn("machine learning", doc.text)
        self.assertNotIn("AI AI AI", doc.text)


class LocalTests(unittest.TestCase):
    def test_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "acme_2023.html").write_text("<p>We use AI.</p>", encoding="utf-8")
            Path(tmp, "notes.docx").write_text("ignored", encoding="utf-8")
            src = LocalSource(tmp)
            filings = src.list_filings()
            self.assertEqual(len(filings), 1)
            self.assertEqual((filings[0].firm_id, filings[0].fiscal_year), ("acme", 2023))
            self.assertIn("We use AI.", src.fetch_document(filings[0]).text)

    def test_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.htm").write_text("<p>x</p>", encoding="utf-8")
            Path(tmp, "b.htm").write_text("<p>y</p>", encoding="utf-8")
            Path(tmp, "meta.csv").write_text(
                "file,firm_id,firm_name,fiscal_year,country,industry_code,industry_scheme\n"
                "a.htm,A1,Alpha plc,2022,gb,7372,SIC\n",
                encoding="utf-8",
            )
            filings = LocalSource(tmp, metadata=Path(tmp, "meta.csv")).list_filings()
            self.assertEqual(len(filings), 1)  # b.htm is not listed in the metadata
            self.assertEqual(filings[0].country, "GB")
            self.assertEqual(filings[0].extra["nace_section"], "J")

    def test_registry(self):
        self.assertIsInstance(get_source("esef", cache_dir=None), XbrlFilingsSource)
        with self.assertRaises(ValueError):
            get_source("nope")

    def test_filing_roundtrip(self):
        f = Filing(source="s", filing_id="i", firm_id="f", firm_name="n", fiscal_year=2020, extra={"k": 1})
        g = Filing.from_dict(f.to_dict())
        self.assertEqual(f, g)


if __name__ == "__main__":
    unittest.main()
