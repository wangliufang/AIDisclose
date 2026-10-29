"""Forms 20-F and 40-F, EDGAR exhibits, country codes and section filters."""

import tempfile
import unittest
from pathlib import Path

from helpers import FakeSession, fixture

from aidisclose import LocalSource, Pipeline
from aidisclose.sources import EdgarSource
from aidisclose.sources.edgar import country_from_edgar, parse_filing_index, schema_label
from aidisclose.text import html_to_text
from aidisclose.text.sections import (
    SECTION_PRESETS,
    SectionFilter,
    is_risk_section,
    section_title,
    split_annual_report,
    split_sections,
)

UA = "Test Researcher test@example.org"
ARCH = "https://www.sec.gov/Archives/edgar/data"


def fpi_session():
    return FakeSession(
        {
            "https://data.sec.gov/submissions/CIK0009999902.json": "edgar_submissions_fpi.json",
            f"{ARCH}/9999902/": "sample_20f.htm",
        }
    )


def ca_session(index="edgar_index_40f.htm"):
    acc = "000999990325000005"
    return FakeSession(
        {
            "https://data.sec.gov/submissions/CIK0009999903.json": "edgar_submissions_ca.json",
            f"{ARCH}/9999903/{acc}/0009999903-25-000005-index.htm": index,
            f"{ARCH}/9999903/{acc}/ex991": "ex991_aif.htm",
            f"{ARCH}/9999903/{acc}/ex992": "ex992_mda.htm",
            f"{ARCH}/9999903/{acc}/ex993": "ex993_fs.htm",
            f"{ARCH}/9999903/{acc}/mrc-40f2024.htm": "sample_40f.htm",
        }
    )


def ex13_session():
    acc = "000999990425000002"
    return FakeSession(
        {
            "https://data.sec.gov/submissions/CIK0009999904.json": "edgar_submissions_ex13.json",
            f"{ARCH}/9999904/{acc}/0009999904-25-000002-index.htm": "edgar_index_ex13.htm",
            f"{ARCH}/9999904/{acc}/lsf-ex13.htm": "ex13_annual_report.htm",
            f"{ARCH}/9999904/{acc}/lsf-20241231.htm": "sample_10k_ex13.htm",
        }
    )


class CountryCodeTests(unittest.TestCase):
    def test_mapping(self):
        cases = {
            "NY": "US",
            "X1": "US",
            "X0": "GB",
            "A6": "CA",
            "Z4": "CA",
            "2M": "DE",
            "F4": "CN",
            "M0": "JP",
            "K3": "HK",
            "L2": "IE",
            "PR": "PR",
            "E9": "KY",
            "D0": "BM",
            "2Q": "GE",
            "GA": "US",
        }
        for code, iso in cases.items():
            self.assertEqual(country_from_edgar(code), iso, code)
        self.assertIsNone(country_from_edgar("XX"))
        self.assertIsNone(country_from_edgar(None))
        self.assertEqual(country_from_edgar("x0"), "GB")

    def test_schema_label(self):
        self.assertEqual(schema_label("10-K405"), "10-K")
        self.assertEqual(schema_label("20-F/A"), "20-F")
        self.assertEqual(schema_label(None), "DOC")


class Form20FTests(unittest.TestCase):
    def test_list_and_fetch(self):
        src = EdgarSource(UA, cache_dir=None, session=fpi_session(), max_rps=10)
        filings = src.list_filings(ciks=[9999902], forms=("20-F",))
        self.assertEqual([f.fiscal_year for f in filings], [2023, 2024])  # 6-K excluded
        f = filings[-1]
        self.assertEqual((f.form, f.country, f.extra["nace_section"]), ("20-F", "GB", "J"))
        doc = src.fetch_document(f)
        for item in ("20-F:cover", "20-F:3", "20-F:4", "20-F:5", "20-F:19"):
            self.assertIn(item, doc.sections)
        self.assertIn("3.D. Risk Factors", doc.sections["20-F:3"])
        self.assertNotIn("index.htm", " ".join(u for u, _ in src.http.session.calls))  # no exhibits for 20-F
        self.assertTrue(is_risk_section("20-F:3"))
        self.assertFalse(is_risk_section("20-F:4"))
        self.assertEqual(section_title("20-F:5"), "Operating and Financial Review and Prospects")

    def test_pipeline_core_preset(self):
        src = EdgarSource(UA, cache_dir=None, session=fpi_session(), max_rps=10)
        f = src.list_filings(ciks=[9999902], forms=("20-F",), start_year=2024)
        res = Pipeline(sections="core").run(f, source=src)
        s = res.sentences
        self.assertTrue(set(s["section"]) <= {"20-F:3", "20-F:4", "20-F:5"})
        llm = s[s["text"].str.startswith("We use large language models")]
        self.assertEqual(llm["state"].tolist(), ["SUBSTANTIVE"])
        self.assertEqual(res.documents["country"].tolist(), ["GB"])


class FilingIndexTests(unittest.TestCase):
    def test_parse(self):
        rows = parse_filing_index(fixture("edgar_index_40f.htm").read_bytes())
        self.assertEqual([r["type"] for r in rows], ["40-F", "EX-99.1", "EX-99.2", "EX-99.3", "EX-23.1", "EX-101.SCH"])
        self.assertTrue(rows[0]["url"].startswith(f"{ARCH}/9999903/"))  # '/ix?doc=' removed
        self.assertEqual(rows[0]["document"], "mrc-40f2024.htm")
        self.assertEqual(rows[1]["description"], "ANNUAL INFORMATION FORM")

    def test_parse_non_index(self):
        self.assertEqual(parse_filing_index(b"<html><body><table><tr><td>x</td></tr></table></body></html>"), [])


class Form40FTests(unittest.TestCase):
    def test_exhibits_labelled(self):
        src = EdgarSource(UA, cache_dir=None, session=ca_session(), max_rps=10)
        f = src.list_filings(ciks=[9999903], forms=("40-F",))[0]
        self.assertEqual((f.form, f.country), ("40-F", "CA"))
        doc = src.fetch_document(f)
        self.assertIn("40-F:AIF", doc.sections)
        self.assertIn("40-F:MDA", doc.sections)
        self.assertFalse(any("EX-99.3" in k or "EX-23" in k for k in doc.sections))
        self.assertIn("40-F:main", doc.sections)
        self.assertEqual([e["section"] for e in f.extra["exhibits"]], ["40-F:AIF", "40-F:MDA"])
        res = Pipeline(sections="core").run([f], source=src)
        s = res.sentences
        self.assertTrue(set(s["section"]) <= {"40-F:AIF", "40-F:MDA", "40-F:main"})
        cv = s[s["text"].str.contains("computer vision")]
        self.assertEqual(cv["state"].tolist(), ["SUBSTANTIVE"])

    def test_exhibits_unlabelled(self):
        src = EdgarSource(UA, cache_dir=None, session=ca_session("edgar_index_unlabelled_40f.htm"), max_rps=10)
        f = src.list_filings(ciks=[9999903], forms=("40-F",))[0]
        doc = src.fetch_document(f)
        self.assertIn("40-F:EX-99.1", doc.sections)
        self.assertIn("40-F:EX-99.2", doc.sections)
        self.assertNotIn("40-F:EX-99.3", doc.sections)  # financial statements skipped

    def test_exhibits_none(self):
        sess = ca_session()
        src = EdgarSource(UA, cache_dir=None, session=sess, max_rps=10, exhibits="none")
        f = src.list_filings(ciks=[9999903], forms=("40-F",))[0]
        doc = src.fetch_document(f)
        self.assertFalse(any("index.htm" in u for u, _ in sess.calls))
        self.assertNotIn("computer vision", doc.text)
        with self.assertRaises(ValueError):
            EdgarSource(UA, cache_dir=None, exhibits="some")


class Exhibit13Tests(unittest.TestCase):
    def test_incorporated_by_reference(self):
        src = EdgarSource(UA, cache_dir=None, session=ex13_session(), max_rps=10)
        f = src.list_filings(ciks=[9999904])[0]
        doc = src.fetch_document(f)
        self.assertIn("10-K:EX-13", doc.sections)
        self.assertIn("10-K:7", doc.sections)
        # Exhibit 13 is selected together with Item 7
        res = Pipeline(sections=["10-K:7"]).run([f], source=src)
        s = res.sentences
        self.assertEqual(set(s["section"]), {"10-K:EX-13"})
        self.assertEqual(s["state"].tolist(), ["SUBSTANTIVE"])
        res_1a = Pipeline(sections=["10-K:1A"]).run([f], source=src)
        self.assertEqual(set(res_1a.sentences["section"]), {"10-K:1A"})

    def test_no_index_request_without_reference(self):
        from test_sources import edgar_session

        sess = edgar_session()
        src = EdgarSource(UA, cache_dir=None, session=sess, max_rps=10)
        f = src.list_filings(ciks=[9999901], start_year=2024)[0]
        src.fetch_document(f)
        self.assertFalse(any("index.htm" in u for u, _ in sess.calls))


class SectionFilterTests(unittest.TestCase):
    def test_semantics(self):
        flt = SectionFilter(["10-K:1A", "10-K:7"])
        self.assertTrue(flt("10-K:1A"))
        self.assertFalse(flt("10-K:8"))
        self.assertTrue(flt("10-K:EX-13"))
        self.assertTrue(flt("10-K:EX-13(2)"))
        self.assertTrue(flt("10-K:main"))
        self.assertTrue(flt("full"))
        self.assertTrue(flt("20-F:8"))  # 20-F not restricted by a 10-K filter
        self.assertTrue(flt("AR:financial_statements"))
        self.assertFalse(SectionFilter(["10-K:1A"])("10-K:EX-13"))

    def test_presets_and_strings(self):
        core = SectionFilter("core")
        self.assertEqual(core.labels, list(SECTION_PRESETS["core"]))
        self.assertFalse(core("AR:financial_statements"))
        self.assertFalse(core("20-F:19"))
        self.assertTrue(core("AR:risk"))
        self.assertEqual(SectionFilter("10-K:1, 10-K:7").labels, ["10-K:1", "10-K:7"])
        self.assertFalse(SectionFilter(None))
        self.assertTrue(SectionFilter(None)("anything"))
        self.assertTrue(SectionFilter("narrative")("AR:sustainability"))


class AnnualReportSplitTests(unittest.TestCase):
    TEXT = "\n".join(
        [
            "Annual Report 2024",
            "Contents",
            "Strategic report 4",
            "Principal risks 20",
            "Strategic report",
            " ".join(["We deployed machine learning models in our warehouses."] * 6),
            "Principal risks and uncertainties",
            " ".join(["Failures of AI systems could disrupt our operations."] * 6),
            "Sustainability statement",
            " ".join(["We report our emissions under the ESRS."] * 6),
            "Independent auditor's report",
            " ".join(["In our opinion the financial statements give a true and fair view."] * 6),
            "Consolidated income statement",
            " ".join(["Revenue 1,234 1,111 cost of sales 800 700 and other figures."] * 6),
        ]
    )

    def test_parts(self):
        parts = split_annual_report(self.TEXT)
        self.assertEqual(
            list(parts),
            ["AR:front", "AR:management", "AR:risk", "AR:sustainability", "AR:auditor", "AR:financial_statements"],
        )
        self.assertIn("Contents", parts["AR:front"])  # TOC lines are too short to count as headings
        self.assertIn("machine learning", parts["AR:management"])
        self.assertTrue(is_risk_section("AR:risk"))
        self.assertEqual(section_title("AR:financial_statements"), "Financial statements")

    def test_fallback(self):
        self.assertEqual(split_annual_report("Just one paragraph of text."), {"full": "Just one paragraph of text."})
        self.assertEqual(list(split_sections(self.TEXT, "ESEF"))[0], "AR:front")
        text = html_to_text(fixture("sample_20f.htm").read_text(encoding="utf-8"))
        self.assertEqual(list(split_sections(text, "20-F"))[0], "20-F:cover")


class LocalFormTests(unittest.TestCase):
    def test_form_from_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "albion_20-F_2024.htm").write_bytes(fixture("sample_20f.htm").read_bytes())
            Path(tmp, "acme-10k-2023.htm").write_text("<p>x</p>", encoding="utf-8")
            Path(tmp, "report_2023.htm").write_text("<p>y</p>", encoding="utf-8")
            src = LocalSource(tmp)
            forms = {f.filing_id.split(":", 1)[1]: f.form for f in src.list_filings()}
            self.assertEqual(
                forms, {"acme-10k-2023.htm": "10-K", "albion_20-F_2024.htm": "20-F", "report_2023.htm": None}
            )
            f20 = next(f for f in src.list_filings() if f.form == "20-F")
            self.assertIn("20-F:4", src.fetch_document(f20).sections)


if __name__ == "__main__":
    unittest.main()
