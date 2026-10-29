import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from aidisclose.cli import _write_report, _write_sample, main


def sample_frames():
    frame = pd.DataFrame([{"sentence_id": "s", "text": "AI text", "gold_state": ""}])
    key = pd.DataFrame([{"sentence_id": "s", "text": "AI text", "sampling_weight": 1}])
    return frame, key


def args(out, key=None, ls=None, results=None):
    return SimpleNamespace(
        out=str(out), key=str(key) if key else None, label_studio=str(ls) if ls else None, results=results or []
    )


def test_annotation_key_collision_preserves_existing_file(tmp_path):
    out = tmp_path / "same.csv"
    out.write_text("original")
    with pytest.raises(SystemExit, match="same file"):
        _write_sample(args(out, out), *sample_frames(), 1)
    assert out.read_text() == "original"


def test_sample_rejects_task_or_config_collision(tmp_path):
    out = tmp_path / "tasks_config.xml"
    out.write_text("original")
    with pytest.raises(SystemExit, match="same file"):
        _write_sample(args(out, ls=tmp_path / "tasks.json"), *sample_frames(), 1)
    assert out.read_text() == "original" and not (tmp_path / "tasks.json").exists()


def test_sample_cannot_overwrite_input_result(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    sent = result / "sentences.csv"
    sent.write_text("source data")
    with pytest.raises(SystemExit, match="result or checkpoint"):
        _write_sample(args(sent, results=[str(result)]), *sample_frames(), 1)
    assert sent.read_text() == "source data"


def test_report_collision_with_json_or_input_preserves_file(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("original")
    with pytest.raises(SystemExit, match="same file"):
        _write_report(target, "markdown", {"n": 1}, default=str)
    assert target.read_text() == "original"
    gold = tmp_path / "gold.md"
    gold.write_text("gold")
    with pytest.raises(SystemExit, match="same file"):
        _write_report(gold, "markdown", {"n": 1}, default=str, inputs=[("gold", gold)])
    assert gold.read_text() == "gold"


def test_output_alias_via_symlink_is_rejected(tmp_path):
    gold = tmp_path / "gold.csv"
    gold.write_text("gold")
    alias = tmp_path / "alias.csv"
    try:
        alias.symlink_to(gold)
    except OSError:
        pytest.skip("symlink creation unavailable on this filesystem")
    with pytest.raises(SystemExit, match="same file"):
        _write_report(alias, "markdown", {}, default=str, inputs=[("gold", gold)])
    assert gold.read_text() == "gold"


def test_agreement_same_input_rejected_and_names_match_disagreements(tmp_path):
    first = tmp_path / "a" / "labels.csv"
    second = tmp_path / "b" / "labels.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("sentence_id,gold_state\ns1,WEAK\ns2,NO_CLAIM\n")
    second.write_text("sentence_id,gold_state\ns1,SUBSTANTIVE\ns2,NO_CLAIM\n")
    with pytest.raises(SystemExit, match="same annotation file"):
        main(["agreement", "--annotations", str(first), str(first)])
    report = tmp_path / "agreement.md"
    main(["agreement", "--annotations", str(first), str(second), "--out", str(report)])
    rep = json.loads(report.with_suffix(".json").read_text())
    assert len(set(rep["annotators"])) == 2
    assert rep["pairs"][0]["a"] == rep["annotators"][0] and rep["pairs"][0]["b"] == rep["annotators"][1]
    assert set(rep["disagreements"][0]) == {"sentence_id", *rep["annotators"]}


def test_training_output_collision_rejected_before_input_read(tmp_path):
    model = tmp_path / "output.joblib"
    model.write_text("existing")
    with pytest.raises(SystemExit, match="same file"):
        main(
            [
                "train",
                "--gold",
                str(tmp_path / "missing.csv"),
                "--model-out",
                str(model),
                "--cv",
                "2",
                "--oof-out",
                str(model),
            ]
        )
    assert model.read_text() == "existing"


def test_output_alias_via_hardlink_is_rejected(tmp_path):
    gold = tmp_path / "gold.csv"
    gold.write_text("gold")
    alias = tmp_path / "alias.csv"
    try:
        os.link(gold, alias)
    except OSError:
        pytest.skip("hard links unavailable on this filesystem")
    with pytest.raises(SystemExit, match="same file"):
        _write_report(alias, "markdown", {}, default=str, inputs=[("gold", gold)])
    assert gold.read_text() == "gold"
