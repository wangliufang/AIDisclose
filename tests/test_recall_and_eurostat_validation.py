import importlib.resources as resources
import json
from pathlib import Path

import pandas as pd
import pytest

from aidisclose import LocalSource, Pipeline
from aidisclose.cli import main
from aidisclose.sources import XbrlFilingsSource
from aidisclose.validate.external import eurostat_ai_share, eurostat_ai_share_by_industry
from aidisclose.validate.recall import found_from_predictions


def test_prediction_counts_only_valid_main_lexicon_matches():
    rows = pd.DataFrame(
        {
            "state": ["SUBSTANTIVE", "WEAK", "WEAK", "WEAK"],
            "raw_terms": ["ai", "ai", "", "ai"],
            "classification_status": ["ok", "ok", "ok", "error"],
        }
    )
    assert found_from_predictions(rows) == 1
    assert found_from_predictions(rows, states=("SUBSTANTIVE", "WEAK")) == 2


def test_cli_results_based_claim_recall_is_computed(tmp_path):
    base = Path(str(resources.files("aidisclose.demo").joinpath("data")))
    src = LocalSource(base, metadata=base / "metadata.csv")
    res = Pipeline(recall_random=3, recall_probe_cap=3).run(src.list_filings(), src)
    result = tmp_path / "res"
    res.save(result)
    sample = tmp_path / "sample.csv"
    report = tmp_path / "report.md"
    main(["sample-recall", "--results", str(result), "--n", "20", "--out", str(sample)])
    gold = pd.read_csv(sample, keep_default_na=False)
    gold["gold_state"] = "NO_CLAIM"
    gold.to_csv(sample, index=False)
    main(
        [
            "evaluate-recall",
            "--gold",
            str(sample),
            "--key",
            str(tmp_path / "sample_key.csv"),
            "--results",
            str(result),
            "--n-boot",
            "3",
            "--out",
            str(report),
        ]
    )
    saved = json.loads(report.with_suffix(".json").read_text())
    assert saved["found_source"] == "unvalidated classifier predictions"
    assert "unvalidated classifier predictions" in report.read_text()
    data = saved["estimates"]
    expected = found_from_predictions(res.sentences, states=("SUBSTANTIVE", "WEAK"))
    assert expected > 0 and data["found_claim"] == expected
    assert data["recall_claim"] == pytest.approx(1.0)


def fake_stat(dims, value=20):
    return {
        "id": list(dims),
        "size": [1] * len(dims),
        "dimension": {k: {"category": {"index": {v: 0}}} for k, v in dims.items()},
        "value": {"0": value},
    }


@pytest.mark.parametrize("missing", ["size_emp", "indic_is", "unit"])
def test_local_country_eurostat_requires_dimensions(tmp_path, missing):
    dims = {
        "size_emp": "GE250",
        "indic_is": "E_AI_TANY",
        "unit": "PC_ENT",
        "geo": "DK",
        "time": "2023",
        "nace_r2": "C10-S951_X_K",
    }
    del dims[missing]
    p = tmp_path / "e.json"
    p.write_text(json.dumps(fake_stat(dims)))
    with pytest.raises(ValueError, match="required dimensions"):
        eurostat_ai_share(path=p)


@pytest.mark.parametrize("missing", ["size_emp", "indic_is", "unit"])
def test_local_industry_eurostat_requires_dimensions(tmp_path, missing):
    dims = {"size_emp": "GE10", "indic_is": "E_AI_TANY", "unit": "PC_ENT", "geo": "DK", "time": "2023", "nace_r2": "C"}
    del dims[missing]
    p = tmp_path / "e.json"
    p.write_text(json.dumps(fake_stat(dims)))
    with pytest.raises(ValueError, match="required dimensions"):
        eurostat_ai_share_by_industry(path=p)


@pytest.mark.parametrize("val", [-1, 101, float("inf")])
def test_eurostat_values_are_plausible_percentages(tmp_path, val):
    dims = {
        "size_emp": "GE250",
        "indic_is": "E_AI_TANY",
        "unit": "PC_ENT",
        "geo": "DK",
        "time": "2023",
        "nace_r2": "C10-S951_X_K",
    }
    p = tmp_path / "e.json"
    p.write_text(json.dumps(fake_stat(dims, val)))
    with pytest.raises(ValueError, match="between 0 and 100"):
        eurostat_ai_share(path=p)


def test_missing_xbrl_identity_cannot_coalesce_firms():
    src = XbrlFilingsSource(cache_dir=None)
    item = {
        "id": "123",
        "attributes": {"fxo_id": "unknown-entry", "report_url": "/r.xhtml", "period_end": "2023-12-31"},
    }
    with pytest.raises(ValueError, match="lacks a usable entity identifier"):
        src._to_filing(item, {})
    item["attributes"]["fxo_id"] = "ABCDEFGHIJKLMNOPQRST-2023-12-31-ESEF-DK-0"
    f = src._to_filing(item, {})
    assert f.firm_id == "lei:ABCDEFGHIJKLMNOPQRST"
    assert f.extra["entity_identity_basis"] == "fxo_id_prefix_unverified"
