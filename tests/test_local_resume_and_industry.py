import json

import pandas as pd
import pytest

from aidisclose import LocalSource, Pipeline
from aidisclose.cli import main
from aidisclose.models import Filing


def test_local_uri_content_is_bound_to_resume(tmp_path):
    path = tmp_path / "report with spaces.txt"
    path.write_text("We use artificial intelligence in our factories.")
    f = Filing("local", "one", "firm", "Firm", 2023, url=path.as_uri(), language="en")
    src = LocalSource(tmp_path)
    pipe = Pipeline()
    ck = tmp_path / "checkpoint"
    first = pipe.run([f], src, checkpoint_dir=ck)
    assert first.sentences.text.str.contains("artificial intelligence").any()
    path.write_text("We use machine learning in our logistics systems.")
    with pytest.raises(ValueError, match="different configuration"):
        pipe.run([f], src, checkpoint_dir=ck, resume=True)
    second = pipe.run([f], src)
    assert first.provenance["input_manifest_sha256"] != second.provenance["input_manifest_sha256"]
    assert second.sentences.text.str.contains("machine learning").any()


def test_cli_filing_url_resume_detects_modified_bytes(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    fpath = root / "r.txt"
    fpath.write_text("We use artificial intelligence in factories.")
    f = Filing("local", "one", "firm", "Firm", 2023, url=fpath.as_uri(), language="en")
    filing_csv = tmp_path / "filings.csv"
    pd.DataFrame([f.to_dict()]).to_csv(filing_csv, index=False)
    out = tmp_path / "out"
    args = ["run", "--source", "local", "--dir", str(root), "--filings", str(filing_csv), "--out", str(out), "-q"]
    main(args)
    fpath.write_text("We use machine learning in factories.")
    with pytest.raises(SystemExit, match="different configuration"):
        main(args + ["--resume"])


def make_industry_fixture(path):
    dims = {"size_emp": "GE10", "indic_is": "E_AI_TANY", "unit": "PC_ENT", "nace_r2": "C", "geo": "DK", "time": "2023"}
    obj = {
        "id": list(dims),
        "size": [1] * len(dims),
        "dimension": {k: {"category": {"index": {v: 0}}} for k, v in dims.items()},
        "value": {"0": 20},
    }
    path.write_text(json.dumps(obj))


def test_industry_flags_cannot_be_ignored(tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("country,nace_section,fiscal_year,n_firms,share_claim\nDK,C,2023,10,0.2\n")
    fixture = tmp_path / "e.json"
    make_industry_fixture(fixture)
    out = tmp_path / "out.csv"
    args = [
        "eurostat",
        "--dataset",
        "isoc_eb_ain2",
        "--panel",
        str(panel),
        "--eurostat-file",
        str(fixture),
        "--out",
        str(out),
    ]
    with pytest.raises(SystemExit, match="limited to GE10"):
        main(args + ["--size", "GE250"])
    with pytest.raises(SystemExit, match="not the industry dataset"):
        main(args + ["--nace-aggregate", "C"])
    assert not out.exists()
    main(args + ["--size", "GE10"])
    assert pd.read_csv(out).eurostat_share.tolist() == [0.2]
