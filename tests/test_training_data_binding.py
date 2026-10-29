import importlib.resources as resources
from pathlib import Path

import pandas as pd
import pytest

from aidisclose import LocalSource, Pipeline
from aidisclose.cli import main
from aidisclose.models import Document, Filing
from aidisclose.pipeline import Results
from aidisclose.validate.sampling import blind, sample_for_annotation, seal_sample_key


def test_nested_concat_retains_every_constituent_provenance():
    p = Pipeline()

    def run(fid):
        return p.run_documents([Document(Filing("local", fid, fid, fid, 2023), "Plain text.", language="en")])

    a, b, c = [run(x) for x in ("a", "b", "c")]
    x = Results.concat([Results.concat([a, b]), c])
    y = Results.concat([a, Results.concat([b, c])])
    digests = [z.provenance["input_manifest_sha256"] for z in (a, b, c)]
    assert [z["input_manifest_sha256"] for z in x.provenance["combined_runs"]] == digests
    assert x.provenance["combined_runs"] == y.provenance["combined_runs"]
    assert x.provenance["n_input_filings"] == 3
    assert "input_manifest_sha256" not in x.provenance


def demo_sample(tmp_path, incomplete=False):
    base = Path(str(resources.files("aidisclose.demo").joinpath("data")))
    src = LocalSource(base, metadata=base / "metadata.csv")
    res = Pipeline().run(src.list_filings(), src)
    if incomplete:
        res.sentences.loc[res.sentences.index[0], "classification_status"] = "error"
    path = tmp_path / "results"
    res.save(path)
    res = Results.load(path)
    sample = sample_for_annotation(res.sentences, n=1000, min_per_stratum=1)
    visible, key = blind(sample)
    key["dataset_sha256"] = res.dataset_digest()
    key["classification_complete"] = "false" if incomplete else "true"
    key = seal_sample_key(key)
    visible["gold_state"] = [" SUBSTANTIVE " if i % 2 else " WEAK " for i in range(len(visible))]
    gold = tmp_path / "gold.csv"
    k = tmp_path / "key.csv"
    visible.to_csv(gold, index=False)
    key.to_csv(k, index=False)
    return res, path, gold, k


def test_train_uses_original_candidates_and_normalized_labels(tmp_path, monkeypatch):
    from aidisclose.classify.supervised import SupervisedClassifier

    res, path, gold, key = demo_sample(tmp_path)
    captured = {}

    class Model:
        def save(self, path):
            Path(path).write_bytes(b"test")

    def fake_train(cands, labels, sample_weight=None):
        captured.update(cands=cands, labels=labels, weights=sample_weight)
        return Model()

    monkeypatch.setattr(SupervisedClassifier, "train", fake_train)
    args = [
        "train",
        "--gold",
        str(gold),
        "--key",
        str(key),
        "--results",
        str(path),
        "--weighted",
        "--model-out",
        str(tmp_path / "m"),
    ]
    main(args)
    original = {c.sentence_id: c for c in Pipeline().candidates_from_results(res)}
    assert len(captured["cands"]) == len(original) < len(res.sentences)
    assert set(captured["labels"]) == {"WEAK", "SUBSTANTIVE"}
    assert len(captured["weights"]) == len(original)
    assert all(
        c.section == original[c.sentence_id].section and c.term_spans == original[c.sentence_id].term_spans
        for c in captured["cands"]
    )
    assert all(c.section != "edited_wrong_section" for c in captured["cands"])
    with pytest.raises(SystemExit, match="requires --results"):
        main(["train", "--gold", str(gold), "--key", str(key), "--model-out", str(tmp_path / "m")])


def test_weighted_train_rejects_incomplete_classification(tmp_path):
    res, path, gold, key = demo_sample(tmp_path, incomplete=True)
    with pytest.raises(SystemExit, match="complete classification"):
        main(
            [
                "train",
                "--gold",
                str(gold),
                "--key",
                str(key),
                "--results",
                str(path),
                "--weighted",
                "--model-out",
                str(tmp_path / "m"),
            ]
        )
    assert not (tmp_path / "m").exists()


def test_training_rejects_other_dataset(tmp_path):
    res, path, gold, key = demo_sample(tmp_path)
    other = Pipeline().run_documents(
        [Document(Filing("local", "other", "other", "Other", 2024), "Machine learning runs.", language="en")]
    )
    otherpath = tmp_path / "other"
    other.save(otherpath)
    with pytest.raises(SystemExit, match="different datasets"):
        main(
            [
                "train",
                "--gold",
                str(gold),
                "--key",
                str(key),
                "--results",
                str(otherpath),
                "--model-out",
                str(tmp_path / "m"),
            ]
        )


def test_supervised_train_and_cv_with_original_candidates(tmp_path):
    from aidisclose.classify.supervised import SupervisedClassifier

    res, path, gold, key = demo_sample(tmp_path)
    model = tmp_path / "model.joblib"
    oof = tmp_path / "oof.csv"
    main(
        [
            "train",
            "--gold",
            str(gold),
            "--key",
            str(key),
            "--results",
            str(path),
            "--weighted",
            "--cv",
            "2",
            "--cv-group-by",
            "firm_id",
            "--n-boot",
            "5",
            "--model-out",
            str(model),
            "--oof-out",
            str(oof),
        ]
    )
    fitted = SupervisedClassifier.load(model)
    assert fitted.meta["n_train"] == len(Pipeline().candidates_from_results(res))
    assert len(pd.read_csv(oof)) == fitted.meta["n_train"]


def test_multikey_evaluate_normalizes_whitespace_labels(tmp_path):
    res, path, gold, key = demo_sample(tmp_path)
    output = tmp_path / "report.md"
    main(
        [
            "evaluate",
            "--gold",
            str(gold),
            "--key",
            str(key),
            "--results",
            str(path),
            "--weighted",
            "--on",
            "sentence_id",
            "firm_id",
            "--cluster-by",
            "none",
            "--n-boot",
            "5",
            "--out",
            str(output),
        ]
    )
    assert output.exists()
