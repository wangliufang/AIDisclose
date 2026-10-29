import pandas as pd
import pytest

from aidisclose import LocalSource
from aidisclose.models import Filing


def make_filing(year):
    return Filing("local", "f", "firm", "Firm", year)


@pytest.mark.parametrize("year", [2023.5, "2023.5", -1, "10000", float("inf"), True, "twenty"])
def test_invalid_fiscal_year_is_never_truncated(year):
    with pytest.raises(ValueError, match="fiscal_year"):
        make_filing(year)
    with pytest.raises(ValueError, match="fiscal_year"):
        Filing.from_dict({"source": "local", "filing_id": "f", "firm_id": "a", "firm_name": "A", "fiscal_year": year})


@pytest.mark.parametrize(
    ("year", "expected"),
    [(None, None), (float("nan"), None), (pd.NA, None), ("", None), ("2023.0", 2023), (2023, 2023)],
)
def test_fiscal_year_round_trip(year, expected):
    assert make_filing(year).fiscal_year == expected
    assert (
        Filing.from_dict(
            {"source": "local", "filing_id": "f", "firm_id": "a", "firm_name": "A", "fiscal_year": year}
        ).fiscal_year
        == expected
    )


def test_metadata_missing_paths_rejected(tmp_path):
    (tmp_path / "a_2023.txt").write_text("A report.")
    meta = tmp_path / "meta.csv"
    meta.write_text("file,firm_id,fiscal_year\na_2023.txt,a,2023\nmissing.txt,b,2023\n")
    with pytest.raises(ValueError, match="unmatched.*missing.txt"):
        LocalSource(tmp_path, metadata=meta).list_filings()


def test_inferred_company_in_different_folders_needs_explicit_identifier(tmp_path):
    for directory, year in [("a", 2023), ("b", 2024)]:
        p = tmp_path / directory
        p.mkdir()
        (p / f"report_{year}.txt").write_text("A report.")
    with pytest.raises(ValueError, match="inferred firm_id.*multiple directories"):
        LocalSource(tmp_path).list_filings()
    meta = tmp_path / "meta.csv"
    meta.write_text("file,firm_id,fiscal_year\na/report_2023.txt,firm-a,2023\nb/report_2024.txt,firm-b,2024\n")
    out = LocalSource(tmp_path, metadata=meta).list_filings()
    assert [(f.firm_id, f.fiscal_year) for f in out] == [("firm-a", 2023), ("firm-b", 2024)]


def test_local_metadata_invalid_year_errors_instead_of_guessing(tmp_path):
    (tmp_path / "a_2023.txt").write_text("A report.")
    meta = tmp_path / "meta.csv"
    meta.write_text("file,firm_id,fiscal_year\na_2023.txt,a,2023.5\n")
    with pytest.raises(ValueError, match="fiscal_year"):
        LocalSource(tmp_path, metadata=meta).list_filings()


def test_identifiers_that_look_like_missing_values_are_preserved():
    f = Filing.from_dict(
        {"source": "local", "filing_id": "nan", "firm_id": "nan", "firm_name": "NA", "fiscal_year": "2023"}
    )
    assert (f.filing_id, f.firm_id, f.firm_name) == ("nan", "nan", "NA")


def test_cli_reports_invalid_metadata_without_creating_results(tmp_path):
    from aidisclose.cli import main

    (tmp_path / "a_2023.txt").write_text("report")
    meta = tmp_path / "meta.csv"
    meta.write_text("file,firm_id,fiscal_year\na_2023.txt,a,2023.5\n")
    with pytest.raises(SystemExit, match="error: invalid fiscal_year"):
        main(
            [
                "run",
                "--source",
                "local",
                "--dir",
                str(tmp_path),
                "--metadata",
                str(meta),
                "--out",
                str(tmp_path / "results"),
            ]
        )
    assert not (tmp_path / "results").exists()
