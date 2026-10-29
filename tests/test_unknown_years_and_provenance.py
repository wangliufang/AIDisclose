import numpy as np
import pandas as pd
import pytest
from test_aggregate import doc

from aidisclose import Pipeline
from aidisclose.aggregate import firm_year_panel
from aidisclose.models import Document, Filing
from aidisclose.pipeline import SENTENCE_COLUMNS, Results
from aidisclose.validate import cohen_kappa


def test_unknown_years_are_never_merged_silently():
    docs = pd.DataFrame(
        [
            doc("a", "firm", None, "US", 100, 1, 1, 1, 0, 0),
            doc("b", "firm", None, "US", 200, 1, 1, 0, 1, 0),
            doc("c", "firm", 2022, "US", 200, 1, 1, 0, 1, 0),
        ]
    )
    r = Results(sentences=pd.DataFrame(columns=SENTENCE_COLUMNS), documents=docs)
    with pytest.raises(ValueError, match="2 processed report"):
        firm_year_panel(r)
    with pytest.warns(UserWarning, match="excluding 2"):
        result = firm_year_panel(r, unknown_years="exclude")
    assert result.filing_id.tolist() == ["c"]
    with pytest.raises(ValueError, match="unknown_years"):
        firm_year_panel(r, unknown_years="invalid")


def test_degenerate_kappa_undefined_even_with_weights():
    assert np.isnan(cohen_kappa(["WEAK", "WEAK"], ["WEAK", "WEAK"]))
    assert np.isnan(cohen_kappa(["WEAK", "WEAK"], ["WEAK", "WEAK"], weights=[2, 3]))
    assert cohen_kappa(["WEAK", "NO_CLAIM"], ["WEAK", "NO_CLAIM"]) == 1


def test_in_memory_provenance_tracks_content_and_does_not_mutate_language():
    def make(text="Artificial intelligence is deployed.", sections=None, language=None):
        f = Filing(source="local", filing_id="a", firm_id="a", firm_name="A", fiscal_year=2023)
        return Document(filing=f, text=text, sections=sections or {}, language=language)

    d = make()
    p = Pipeline()
    a = p.run_documents([d]).provenance
    assert d.language is None
    assert p.run_documents([d]).provenance["input_manifest_sha256"] == a["input_manifest_sha256"]
    assert len(a["input_manifest_sha256"]) == 64 and a["n_input_filings"] == 1
    assert a["source"]["name"] == "in-memory-documents"
    assert p.run_documents([make(text="changed")]).provenance["input_manifest_sha256"] != a["input_manifest_sha256"]
    assert (
        p.run_documents([make(sections={"x": "different"})]).provenance["input_manifest_sha256"]
        != a["input_manifest_sha256"]
    )
    assert p.run_documents([make(language="en")]).provenance["input_manifest_sha256"] != a["input_manifest_sha256"]


def test_concat_does_not_attribute_first_digest_to_all_inputs():
    p = Pipeline()

    def run(fid):
        return p.run_documents([Document(Filing("local", fid, fid, fid, 2023), "A plain report.", language="en")])

    a, b = run("a"), run("b")
    result = Results.concat([a, b])
    assert "input_manifest_sha256" not in result.provenance
    assert result.provenance["source"]["name"] == "combined-runs"
    assert result.provenance["n_input_filings"] == 2
    assert [x["input_manifest_sha256"] for x in result.provenance["combined_runs"]] == [
        a.provenance["input_manifest_sha256"],
        b.provenance["input_manifest_sha256"],
    ]


def test_excluding_all_unknown_years_fails_without_writing_empty_panel():
    docs = pd.DataFrame([doc("a", "firm", None, "US", 100, 1, 1, 1, 0, 0)])
    r = Results(sentences=pd.DataFrame(columns=SENTENCE_COLUMNS), documents=docs)
    with pytest.warns(UserWarning, match="excluding 1"):
        with pytest.raises(ValueError, match="no processed reports"):
            firm_year_panel(r, unknown_years="exclude")


def test_panel_cli_requires_explicit_unknown_year_exclusion(tmp_path):
    from aidisclose.cli import main

    p = Pipeline()
    documents = [
        Document(Filing("local", "a", "firm", "Firm", None), "A plain report.", language="en"),
        Document(Filing("local", "b", "firm", "Firm", 2023), "A plain report.", language="en"),
    ]
    results = tmp_path / "result"
    p.run_documents(documents).save(results)
    output = tmp_path / "panel.csv"
    arguments = ["panel", "--results", str(results), "--out", str(output), "--quiet"]
    with pytest.raises(SystemExit, match="unknown fiscal year"):
        main(arguments)
    assert not output.exists()
    with pytest.warns(UserWarning, match="excluding 1"):
        assert main(arguments + ["--exclude-unknown-years"]) in (None, 0)
    assert pd.read_csv(output).filing_id.tolist() == ["b"]
