import numpy as np
import pandas as pd
import pytest

from aidisclose import Pipeline
from aidisclose.models import Document, Filing
from aidisclose.pipeline import Results
from aidisclose.validate.external import compare_with_external, correlation, load_benchmark, permutation_pvalue


def test_relabelled_compatible_runs_can_be_combined_despite_timestamps():
    p = Pipeline()

    def relabel(fid, time):
        d = Document(
            Filing("local", fid, fid, fid, 2023), "We deploy artificial intelligence in logistics.", language="en"
        )
        r = p.relabel(p.run_documents([d]))
        r.provenance["relabelled_utc"] = time
        return r

    a = relabel("a", "2025-01-01T00:00:00Z")
    b = relabel("b", "2025-01-02T00:00:00Z")
    r = Results.concat([a, b])
    assert len(r.documents) == 2
    b.provenance["classifier"] = {"backend": "different"}
    with pytest.raises(ValueError, match="different measurement configurations"):
        Results.concat([a, b])


def test_benchmark_rejects_malformed_values_but_accepts_missing(tmp_path):
    p = tmp_path / "benchmark.csv"
    p.write_text("country,year,value\nUS,2023,12%\nGB,2023,oops\n")
    with pytest.raises(ValueError, match="invalid or nonfinite"):
        load_benchmark(p, value="value")
    p.write_text("country,year,value\nUS,2023,12%\nGB,2023,NA\nDE,2023,\n")
    result = load_benchmark(p, value="value")
    assert result.country.tolist() == ["US"]
    assert result.benchmark_share.iloc[0] == pytest.approx(0.12)


def sample_frames():
    return (
        pd.DataFrame({"country": ["US", "GB", "DE"], "fiscal_year": [2023] * 3, "share_claim": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"country": ["US", "GB", "DE"], "year": [2023] * 3, "eurostat_share": [0.2, 0.3, 0.4]}),
    )


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), "invalid"])
def test_comparison_rejects_bad_group_values(bad):
    g, e = sample_frames()
    g["share_claim"] = g["share_claim"].astype(object)
    g.loc[0, "share_claim"] = bad
    with pytest.raises(ValueError, match="finite numbers"):
        compare_with_external(g, e, measures=["share_claim"])


def test_comparison_accepts_missing_values_and_counts_complete_pairs():
    g, e = sample_frames()
    g.loc[0, "share_claim"] = np.nan
    _, summary = compare_with_external(g, e, measures=["share_claim"])
    assert summary.n_pairs.iloc[0] == 2


def test_correlations_validate_infinity_shapes_and_method():
    assert correlation([1, 2, 3], [2, 4, 6], method="pearson") == pytest.approx(1)
    assert correlation([1, 2, np.nan, 3], [2, 4, 6, 6]) == pytest.approx(1)
    with pytest.raises(ValueError, match="infinite"):
        correlation([1, 2, float("inf")], [2, 4, 6])
    with pytest.raises(ValueError, match="method"):
        permutation_pvalue([1, 2, 3], [3, 2, 1], method="sperman")
    with pytest.raises(ValueError, match="same length"):
        correlation([1, 2], [1])
    with pytest.raises(ValueError, match="n_perm"):
        permutation_pvalue([1, 2, 3], [1, 2, 3], n_perm=0)


def test_cli_benchmark_rejects_invalid_comparison_input(tmp_path):
    from aidisclose.cli import main

    panel = tmp_path / "panel.csv"
    bench = tmp_path / "bench.csv"
    out = tmp_path / "out.csv"
    panel.write_text("country,fiscal_year,n_firms,share_claim\nUS,2023,7,inf\n")
    bench.write_text("country,year,value\nUS,2023,0.2\n")
    with pytest.raises(SystemExit, match="finite numbers"):
        main(["benchmark", "--panel", str(panel), "--file", str(bench), "--value", "value", "--out", str(out)])
    assert not out.exists()


def test_external_values_validated_even_when_unmatched():
    g, e = sample_frames()
    e.loc[len(e)] = ["FR", 2023, float("inf")]
    with pytest.raises(ValueError, match="finite numbers"):
        compare_with_external(g, e, measures=["share_claim"])


@pytest.mark.parametrize("missing", [None, np.nan, ""])
def test_block_permutation_rejects_missing_identifiers(missing):
    with pytest.raises(ValueError, match="blocks cannot contain missing"):
        permutation_pvalue([1, 2, 3, 4], [2, 3, 4, 5], blocks=["A", "A", "B", missing])
