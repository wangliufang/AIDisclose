import pandas as pd
import pytest

from aidisclose.cli import _attach_key
from aidisclose.validate.annotation import from_label_studio, to_label_studio
from aidisclose.validate.sampling import _key_digest, merge_sample_key, seal_sample_key, verify_sample_key


def example():
    key = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "text": "Our machine learning system sorts orders.",
                "firm_name": "Acme",
                "fiscal_year": "2023",
                "section": "10-K:1",
                "prev_text": "We deploy tools.",
                "next_text": "We invest further.",
                "filing_id": "f1",
                "firm_id": "a",
                "sampling_weight": "2",
                "sampling_stratum": "x",
                "dataset_sha256": "abc",
                "classification_complete": "true",
            }
        ]
    )
    visible = key[["sentence_id", "text", "firm_name", "fiscal_year", "section", "prev_text", "next_text"]].copy()
    visible["gold_state"] = "SUBSTANTIVE"
    return visible, seal_sample_key(key)


def test_v2_context_is_sealed_and_legacy_is_readable():
    visible, key = example()
    verify_sample_key(key)
    assert key.sample_schema.eq("2").all()
    changed = key.copy()
    changed.loc[0, "prev_text"] = "Different context."
    with pytest.raises(ValueError, match="contents differ"):
        verify_sample_key(changed)
    legacy = key.drop(columns=["sample_schema", "sample_sha256"]).copy()
    legacy["sample_sha256"] = _key_digest(legacy)
    verify_sample_key(legacy)
    downgraded = key.drop(columns=["sample_schema"])
    with pytest.raises(ValueError, match="contents differ"):
        verify_sample_key(downgraded)


@pytest.mark.parametrize("field", ["text", "firm_name", "fiscal_year", "section", "prev_text", "next_text"])
def test_context_changes_are_rejected(field):
    visible, key = example()
    visible.loc[0, field] = "edited context"
    with pytest.raises(ValueError, match="annotation .* differs"):
        merge_sample_key(visible, key)


def test_cli_key_rejects_changed_context_and_warns_if_missing(tmp_path, capsys):
    visible, key = example()
    p = tmp_path / "key.csv"
    key.to_csv(p, index=False)
    altered = visible.copy()
    altered.loc[0, "next_text"] = "changed"
    with pytest.raises(SystemExit, match="annotation next_text differs"):
        _attach_key(altered, str(p))
    missing = visible.drop(columns=["prev_text"])
    out = _attach_key(missing, str(p))
    assert out.prev_text.tolist() == ["We deploy tools."]
    assert "could not be verified" in capsys.readouterr().err


def test_label_studio_import_preserves_annotated_context():
    visible, key = example()
    task = to_label_studio(visible)[0]
    task["annotations"] = [
        {"completed_by": 1, "result": [{"from_name": "state", "value": {"choices": ["SUBSTANTIVE"]}}]}
    ]
    gold = from_label_studio([task])
    assert gold.loc[0, "prev_text"] == "We deploy tools."
    assert gold.loc[0, "text"] == visible.loc[0, "text"]
    merge_sample_key(gold, key)
    task["data"]["prev_text"] = "Changed."
    with pytest.raises(ValueError, match="prev_text"):
        merge_sample_key(from_label_studio([task]), key)


def test_label_studio_export_handles_nullable_missing_year():
    visible, _ = example()
    visible["fiscal_year"] = pd.array([pd.NA], dtype="Int64")
    task = to_label_studio(visible)[0]
    assert task["data"]["fiscal_year"] == ""
