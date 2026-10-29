import pandas as pd
import pytest
from test_aggregate import doc

from aidisclose.aggregate import firm_year_panel
from aidisclose.cli import main
from aidisclose.pipeline import SENTENCE_COLUMNS, Results


def xbrl_result(periods):
    rows = []
    for i, period in enumerate(periods):
        row = doc(f"x{i}", "firm", 2023, "DK", 100 + i, 1, 1, 0, 1, 0)
        row.update(source="xbrl", period_end=period)
        rows.append(row)
    return Results(sentences=pd.DataFrame(columns=SENTENCE_COLUMNS), documents=pd.DataFrame(rows))


def test_xbrl_periods_must_be_reviewed():
    res = xbrl_result(["2023-12-31"])
    with pytest.raises(ValueError, match="periods are unverified"):
        firm_year_panel(res)
    panel = firm_year_panel(res, xbrl_periods="reviewed")
    assert panel.filing_id.tolist() == ["x0"]
    assert panel.period_end.tolist() == ["2023-12-31"]


def test_multiple_xbrl_periods_never_selected_by_length():
    res = xbrl_result(["2023-06-30", "2023-12-31"])
    with pytest.raises(ValueError, match="multiple XBRL period ends"):
        firm_year_panel(res, xbrl_periods="reviewed")


def test_invalid_xbrl_period_and_option_rejected():
    res = xbrl_result(["not-a-date"])
    with pytest.raises(ValueError, match="valid period_end"):
        firm_year_panel(res, xbrl_periods="reviewed")
    with pytest.raises(ValueError, match="xbrl_periods must"):
        firm_year_panel(res, xbrl_periods="yes")


def test_cli_requires_confirmation_and_refuses_period_conflicts(tmp_path):
    res = xbrl_result(["2023-06-30", "2023-12-31"])
    source = tmp_path / "results"
    res.save(source)
    out = tmp_path / "panel.csv"
    args = ["panel", "--results", str(source), "--out", str(out), "--quiet"]
    with pytest.raises(SystemExit, match="periods are unverified"):
        main(args)
    with pytest.raises(SystemExit, match="multiple XBRL period ends"):
        main(args + ["--confirm-xbrl-periods"])
    assert not out.exists()
