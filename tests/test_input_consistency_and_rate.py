import pytest

import aidisclose.pipeline as pipeline_module
from aidisclose import LocalSource, Pipeline
from aidisclose.models import Filing
from aidisclose.sources import XbrlFilingsSource
from aidisclose.sources.base import HttpClient


@pytest.mark.parametrize(("workers", "executor"), [(1, "thread"), (2, "thread"), (2, "process")])
def test_local_read_rejects_change_after_manifest(tmp_path, monkeypatch, workers, executor):
    for n in ("a", "b"):
        (tmp_path / f"{n}_2023.txt").write_text("We use artificial intelligence in our factories.")
    source = LocalSource(tmp_path)
    filings = source.list_filings()
    for f in filings:
        f.language = "en"
    original = pipeline_module._filing_manifest

    def change_after_hash(*args, **kwargs):
        manifest = original(*args, **kwargs)
        (tmp_path / "a_2023.txt").write_text("We use machine learning in our factories.")
        return manifest

    monkeypatch.setattr(pipeline_module, "_filing_manifest", change_after_hash)
    result = Pipeline().run(filings, source, workers=workers, executor=executor, checkpoint_dir=tmp_path / "ck")
    docs = result.documents.set_index("firm_id")
    assert docs.loc["a", "status"] == "error" and "changed after the run manifest" in docs.loc["a", "error"]
    assert docs.loc["b", "status"] == "ok"
    assert set(result.sentences.firm_id) == {"b"}
    assert result.provenance["source"]["input_content_verified"] is True


def test_appearing_file_does_not_bypass_missing_manifest(tmp_path, monkeypatch):
    path = tmp_path / "new.txt"
    f = Filing("local", "f", "a", "A", 2023, url=path.as_uri(), language="en")
    src = LocalSource(tmp_path)
    original = pipeline_module._filing_manifest

    def appears(*args, **kwargs):
        manifest = original(*args, **kwargs)
        path.write_text("We use artificial intelligence in factories.")
        return manifest

    monkeypatch.setattr(pipeline_module, "_filing_manifest", appears)
    result = Pipeline().run([f], src)
    assert result.documents.status.tolist() == ["error"]
    assert "unavailable when the run manifest" in result.documents.error.iloc[0]


def test_custom_local_subclass_fetch_is_not_bypassed(tmp_path):
    class Custom(LocalSource):
        def fetch_document(self, filing):
            doc = super().fetch_document(filing)
            doc.text = "We use machine learning in our factories."
            return doc

    p = tmp_path / "a_2023.txt"
    p.write_text("We use artificial intelligence in our factories.")
    src = Custom(tmp_path)
    f = src.list_filings()[0]
    f.language = "en"
    result = Pipeline().run([f], src)
    assert result.sentences.text.str.contains("machine learning").any()
    assert "input_content_verified" not in result.provenance["source"]


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), 1e-320, True, "bad"])
def test_rate_cannot_disable_throttling(rate):
    with pytest.raises(ValueError, match="request rate"):
        HttpClient("researcher", max_rps=rate, cache_dir=None)
    with pytest.raises(ValueError, match="request rate"):
        XbrlFilingsSource(max_rps=rate, cache_dir=None)
