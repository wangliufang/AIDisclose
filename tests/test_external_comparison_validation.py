import json

import pandas as pd
import pytest

from aidisclose.sources import XbrlFilingsSource
from aidisclose.validate.external import eurostat_ai_share
from aidisclose.validate.recall import estimate_recall, recall_to_markdown


def stat(nace):
    dims = {
        "size_emp": "GE250",
        "indic_is": "E_AI_TANY",
        "unit": "PC_ENT",
        "nace_r2": nace,
        "geo": "DK",
        "time": "2023",
    }
    return {
        "id": list(dims),
        "size": [1] * len(dims),
        "dimension": {k: {"category": {"index": {v: 0}}} for k, v in dims.items()},
        "value": {"0": 20},
    }


def test_country_comparison_never_defaults_to_single_industry(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps(stat("C")))
    assert eurostat_ai_share(path=p).empty
    df = eurostat_ai_share(path=p, nace_r2="C")
    assert df.nace_r2.tolist() == ["C"] and df.eurostat_share.tolist() == [0.2]
    with pytest.raises(ValueError, match="explicit"):
        eurostat_ai_share(path=p, nace_r2=None)


def test_local_country_eurostat_requires_activity_dimension(tmp_path):
    obj = stat("C10-S951_X_K")
    obj["id"].remove("nace_r2")
    obj["size"].pop(3)
    del obj["dimension"]["nace_r2"]
    p = tmp_path / "e.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="nace_r2"):
        eurostat_ai_share(path=p)


def test_count_based_recall_identifies_predictions_and_rejects_impossible_counts():
    gold = pd.DataFrame(
        {"gold_state": ["WEAK", "NO_CLAIM"], "recall_weight": [1, 1], "recall_stratum": ["random", "random"]}
    )
    rep = estimate_recall(
        gold, found_substantive=1, found_claim=2, n_boot=2, found_source="unvalidated classifier predictions"
    )
    assert rep["found_source"] == "unvalidated classifier predictions"
    assert "unvalidated classifier predictions" in recall_to_markdown(rep)
    with pytest.raises(ValueError, match="cannot be smaller"):
        estimate_recall(gold, found_substantive=2, found_claim=1, n_boot=2)


def test_xbrl_conflicting_entity_identifiers_rejected():
    src = XbrlFilingsSource(cache_dir=None)
    item = {
        "id": "1",
        "attributes": {"fxo_id": "ABCDEFGHIJKLMNOPQRST-2023-12-31-ESEF-DK-0", "report_url": "/r.xhtml"},
        "relationships": {"entity": {"data": {"id": "e"}}},
    }
    with pytest.raises(ValueError, match="conflicting entity identifiers"):
        src._to_filing(item, {"e": {"identifier": "12345678901234567890"}})
    f = src._to_filing(item, {"e": {"identifier": "abcdefghijklmnopqrst"}})
    assert f.extra["entity_identity_basis"] == "entity_identifier"
    assert f.firm_id == "lei:ABCDEFGHIJKLMNOPQRST"
