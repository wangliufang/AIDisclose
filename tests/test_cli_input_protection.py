import os

import pytest

from aidisclose.cli import main


def rejected(args, target, text, match="same file"):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        main(args)
    assert target.read_text(encoding="utf-8") == text


def test_panel_cannot_overwrite_result(tmp_path):
    result = tmp_path / "result"
    target = result / "sentences.csv"
    rejected(["panel", "--results", str(result), "--out", str(target)], target, "source data", "result or checkpoint")


def test_panel_cannot_overwrite_attributes(tmp_path):
    target = tmp_path / "attributes.csv"
    rejected(
        ["panel", "--results", str(tmp_path / "missing"), "--firm-attributes", str(target), "--out", str(target)],
        target,
        "attributes",
    )


def test_relabel_refuses_same_directory_and_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "sentences.csv"
    sentinel.write_text("original")
    alias = tmp_path / "alias"
    destinations = [source]
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError:
        pass  # Symlink creation may require privileges on Windows.
    else:
        destinations.append(alias)
    for dest in destinations:
        with pytest.raises(SystemExit, match="output directory must differ"):
            main(["relabel", "--results", str(source), "--out", str(dest)])
        assert sentinel.read_text() == "original"


@pytest.mark.parametrize("command,flag", [("benchmark", "--file"), ("eurostat", "--eurostat-file")])
def test_comparison_cannot_overwrite_external_input(tmp_path, command, flag):
    target = tmp_path / "external.csv"
    extra = ["--value", "value"] if command == "benchmark" else []
    rejected(
        [command, "--panel", str(tmp_path / "missing"), flag, str(target), "--out", str(target), *extra],
        target,
        "external",
    )


def test_import_cannot_overwrite_export(tmp_path):
    target = tmp_path / "export.json"
    rejected(["import-label-studio", "--export", str(target), "--out", str(target)], target, "[]")


def test_list_cannot_overwrite_local_report(tmp_path):
    target = tmp_path / "firm_2023.txt"
    rejected(
        ["list", "--source", "local", "--dir", str(tmp_path), "--out", str(target)],
        target,
        "The company uses artificial intelligence.",
    )


def test_run_cannot_overwrite_metadata_before_checkpoint(tmp_path):
    out = tmp_path / "results"
    target = out / "documents.csv"
    rejected(
        ["run", "--source", "local", "--dir", str(tmp_path), "--metadata", str(target), "--out", str(out)],
        target,
        "file\nreport.txt\n",
    )
    assert not (out / "checkpoint").exists()


def test_run_rejects_checkpoint_input(tmp_path):
    out = tmp_path / "results"
    target = out / "checkpoint" / "list.csv"
    rejected(
        ["run", "--source", "local", "--dir", str(tmp_path), "--filings", str(target), "--out", str(out)],
        target,
        "anything",
        "inside the checkpoint",
    )


def test_weighted_evaluation_requires_sentence_id(tmp_path):
    with pytest.raises(SystemExit, match="requires sentence_id"):
        main(
            [
                "evaluate",
                "--gold",
                str(tmp_path / "gold"),
                "--key",
                str(tmp_path / "key"),
                "--results",
                str(tmp_path / "result"),
                "--weighted",
                "--on",
                "country",
            ]
        )


def test_run_cannot_overwrite_url_only_report(tmp_path):
    import pandas as pd

    from aidisclose.models import Filing

    out = tmp_path / "results"
    out.mkdir()
    report = out / "documents.csv"
    report.write_text("original report")
    listing = tmp_path / "filings.csv"
    filing = Filing("local", "f1", "firm", "Firm", 2023, url=report.as_uri(), language="en")
    pd.DataFrame([filing.to_dict()]).to_csv(listing, index=False)
    with pytest.raises(SystemExit, match="same file"):
        main(["run", "--source", "local", "--dir", str(tmp_path), "--filings", str(listing), "--out", str(out)])
    assert report.read_text() == "original report"
    assert not (out / "checkpoint").exists()


def test_panel_cannot_overwrite_checkpoint_through_hard_link(tmp_path):
    result = tmp_path / "result"
    checkpoint = result / "checkpoint" / "parts" / "part.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("original checkpoint")
    alias = tmp_path / "alias.csv"
    try:
        os.link(checkpoint, alias)
    except OSError:
        pytest.skip("hard links unavailable on this filesystem")
    with pytest.raises(SystemExit, match="result or checkpoint"):
        main(["panel", "--results", str(result), "--out", str(alias)])
    assert checkpoint.read_text() == "original checkpoint"
