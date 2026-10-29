"""Regressions for source selection, fiscal dates and literal identifiers."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
from helpers import FakeSession

import aidisclose
from aidisclose import Document, Filing, Pipeline
from aidisclose.cli import _list, build_parser, main
from aidisclose.sources import EdgarSource, XbrlFilingsSource


def xbrl_source():
    return XbrlFilingsSource(
        cache_dir=None,
        max_rps=1000,
        session=FakeSession(
            {
                "https://filings.xbrl.org/api/filings?page%5Bnumber%5D=2": "xbrl_filings_page2.json",
                "https://filings.xbrl.org/api/filings": "xbrl_filings_page1.json",
            }
        ),
    )


class TestFeatureIntegrity(unittest.TestCase):
    def test_installed_package_when_requested(self):
        root = os.environ.get("AIDISCLOSE_EXPECT_INSTALL_ROOT")
        if root:
            self.assertTrue(Path(aidisclose.__file__).resolve().is_relative_to(Path(root).resolve()))

    def test_invalid_period_is_not_replaced_by_filing_year(self):
        recent = {
            "accessionNumber": ["a", "b", "c"],
            "form": ["10-K"] * 3,
            "reportDate": ["2024-02-31", "", ""],
            "filingDate": ["2025-03-01", "2025-03-02", "2025-99-99"],
            "primaryDocument": ["x", "y", "z"],
        }
        payload = {"name": "X", "filings": {"recent": recent, "files": []}}
        src = EdgarSource(
            "Tester tester@example.org",
            cache_dir=None,
            max_rps=10,
            session=FakeSession({"https://data.sec.gov/submissions/CIK0000000001.json": json.dumps(payload).encode()}),
        )
        out = {f.extra["accession"]: f for f in src.list_filings(ciks=[1])}
        self.assertEqual([out[k].fiscal_year for k in ("a", "b", "c")], [None, 2024, None])
        self.assertEqual(
            [out[k].extra["fiscal_year_basis"] for k in ("a", "b", "c")],
            ["invalid_period_end", "filing_year_minus_one_estimate", "missing"],
        )

    def test_unknown_periods_are_not_merged(self):
        filings = [
            Filing(source="xbrl", filing_id="a", firm_id="z", firm_name="Z", period_end=None),
            Filing(source="xbrl", filing_id="b", firm_id="z", firm_name="Z", period_end=None),
            Filing(source="xbrl", filing_id="c", firm_id="z", firm_name="Z", period_end="2024-02-31"),
        ]
        self.assertEqual(len(XbrlFilingsSource._dedupe(filings)), 3)

    def test_sec_rate_cannot_disable_throttling(self):
        for rate in (0, -1, float("nan"), float("inf"), 11):
            with self.assertRaisesRegex(ValueError, "request rate"):
                EdgarSource("Tester tester@example.org", cache_dir=None, max_rps=rate)

    def test_country_iterator_and_record_identifiers(self):
        s = xbrl_source()
        out = s.list_filings(countries=iter(["DK", "GB"]), start_year=2023, end_year=2023)
        self.assertEqual(len(out), 2)
        versions = xbrl_source().list_filings(english_only=False, keep_all_versions=True, max_pages=1)
        self.assertEqual(len(versions), 3)
        self.assertEqual(len({f.filing_id for f in versions}), 3)
        self.assertEqual(versions[0].extra["fxo_id"], versions[1].extra["fxo_id"])

    def test_cli_preserves_requested_language_before_deduplication(self):
        args = build_parser().parse_args(["run", "--source", "xbrl", "--languages", "da", "--out", "unused"])
        selected = _list(args, xbrl_source())
        self.assertTrue(any(f.firm_name.startswith("Vantage") and f.language == "da" for f in selected))
        self.assertTrue(any(f.firm_name.startswith("Fjord") and f.language == "da" for f in selected))
        self.assertFalse(any(f.language == "en" for f in selected))
        args = build_parser().parse_args(["list", "--source", "xbrl", "--all-languages", "--out", "unused"])
        self.assertTrue(any(f.language == "da" for f in _list(args, xbrl_source())))

    def test_cli_firm_attribute_id_na_is_literal(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            filing = Filing(source="local", filing_id="one", firm_id="NA", firm_name="Acme", fiscal_year=2024)
            text = "We use machine learning to forecast demand at our stores in the United States."
            Pipeline().run_documents([Document(filing=filing, language="en", text=text)]).save(d / "result")
            (d / "attributes.csv").write_text("firm_id,employees\nNA,42\n")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(
                    [
                        "panel",
                        "--results",
                        str(d / "result"),
                        "--level",
                        "firm-year",
                        "--firm-attributes",
                        str(d / "attributes.csv"),
                        "--out",
                        str(d / "panel.csv"),
                    ]
                )
            out = pd.read_csv(d / "panel.csv", keep_default_na=False)
            self.assertEqual(out.firm_id.iloc[0], "NA")
            self.assertEqual(out.employees.iloc[0], 42)
