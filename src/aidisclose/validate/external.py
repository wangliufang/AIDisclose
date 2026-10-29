"""External validation against official statistics.

The main benchmark is Eurostat's survey on ICT usage in enterprises, which
reports the share of enterprises using at least one AI technology by country,
size class and NACE activity (datasets ``isoc_eb_ai`` and ``isoc_eb_ain2``).
Listed companies are large, so the natural comparison is the size class
``GE250`` (250 persons employed or more).

The module parses Eurostat's JSON-stat 2.0 responses (either downloaded by the
user or fetched through the Eurostat dissemination API) and computes rank and
linear correlations between AIDisclose group-level indicators and the official
shares.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
# ISO 3166-1 alpha-2 -> Eurostat geo codes where they differ
ISO_TO_EUROSTAT = {"GR": "EL", "GB": "UK"}
EUROSTAT_TO_ISO = {v: k for k, v in ISO_TO_EUROSTAT.items()}


def parse_jsonstat(obj: Mapping[str, Any] | str | bytes) -> pd.DataFrame:
    """Convert a JSON-stat 2.0 dataset into a tidy DataFrame.

    One column per dimension (category codes), plus ``<dim>_label`` columns
    and a ``value`` column. Missing cells are omitted.
    """
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    ids: list[str] = list(obj["id"])
    sizes: list[int] = list(obj["size"])
    dims = obj["dimension"]
    codes: list[list[str]] = []
    labels: list[dict[str, str]] = []
    for d in ids:
        cat = dims[d]["category"]
        index = cat.get("index")
        if isinstance(index, dict):
            ordered = sorted(index, key=lambda k: index[k])
        elif isinstance(index, list):
            ordered = list(index)
        else:  # single category without index
            ordered = list(cat.get("label", {}).keys())
        codes.append(ordered)
        labels.append(cat.get("label", {}))
    values = obj.get("value", {})
    if isinstance(values, list):
        items = [(i, v) for i, v in enumerate(values) if v is not None]
    else:
        items = [(int(k), v) for k, v in values.items() if v is not None]
    strides = [int(np.prod(sizes[i + 1 :])) for i in range(len(sizes))]
    rows = []
    for flat, val in items:
        row: dict[str, Any] = {}
        rem = flat
        for d, stride, cl, lab in zip(ids, strides, codes, labels):
            pos, rem = divmod(rem, stride)
            code = cl[pos]
            row[d] = code
            row[d + "_label"] = lab.get(code, code)
        row["value"] = float(val)
        rows.append(row)
    columns = [column for d in ids for column in (d, d + "_label")] + ["value"]
    return pd.DataFrame(rows, columns=columns)


def eurostat_url(
    dataset: str, filters: Mapping[str, str | Sequence[str]] | None = None
) -> tuple[str, list[tuple[str, str]]]:
    params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN")]
    for k, v in (filters or {}).items():
        for item in [v] if isinstance(v, str) else list(v):
            params.append((k, str(item)))
    return EUROSTAT_API.format(dataset=dataset), params


def load_eurostat(
    dataset: str = "isoc_eb_ai",
    filters: Mapping[str, str | Sequence[str]] | None = None,
    path: str | Path | None = None,
    session=None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Load a Eurostat dataset from a local JSON-stat file or from the API."""
    if path is not None:
        return parse_jsonstat(Path(path).read_bytes())
    import requests

    url, params = eurostat_url(dataset, filters)
    sess = session or requests.Session()
    resp = sess.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return parse_jsonstat(resp.content)


def _validate_eurostat_values(df: pd.DataFrame) -> None:
    values = pd.to_numeric(df["value"], errors="coerce")
    if values.isna().any() or (~np.isfinite(values)).any() or ((values < 0) | (values > 100)).any():
        raise ValueError("Eurostat percentage values must be finite and between 0 and 100")


def eurostat_ai_share(
    path: str | Path | None = None,
    size_emp: str = "GE250",
    indic_is: str = "E_AI_TANY",
    unit: str = "PC_ENT",
    nace_r2: str | None = "C10-S951_X_K",
    session=None,
) -> pd.DataFrame:
    """Share of enterprises using at least one AI technology, by country and year.

    Returns ``country`` (ISO alpha-2), ``year``, ``eurostat_share`` (a fraction)
    and the selected ``nace_r2``. The default selects the explicit activity
    aggregate ``C10-S951_X_K``; choose a different aggregate deliberately.
    """
    if not nace_r2 or not str(nace_r2).strip():
        raise ValueError("nace_r2 must name an explicit Eurostat activity aggregate")
    filters = {"size_emp": size_emp, "indic_is": indic_is, "unit": unit, "nace_r2": nace_r2}
    df = load_eurostat("isoc_eb_ai", filters=None if path else filters, path=path, session=session)
    if path is not None:
        missing = [col for col in filters if col not in df.columns]
        if missing:
            raise ValueError(f"local Eurostat data lacks required dimensions: {missing}")
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]
    _validate_eurostat_values(df)
    if "nace_r2" in df.columns and nace_r2 is None and df["nace_r2"].nunique() > 1:
        raise ValueError("several NACE aggregates present; pass nace_r2=...")
    out = pd.DataFrame(
        {
            "country": df["geo"].map(lambda g: EUROSTAT_TO_ISO.get(g, g)),
            "year": pd.to_numeric(df["time"], errors="coerce").astype("Int64"),
            "eurostat_share": df["value"] / 100.0,
            "nace_r2": str(nace_r2),
        }
    )
    # drop EU aggregates such as EU27_2020
    out = out[out["country"].str.len() == 2].reset_index(drop=True)
    return out


def eurostat_ai_share_by_industry(
    path: str | Path | None = None,
    indic_is: str = "E_AI_TANY",
    unit: str = "PC_ENT",
    countries: Sequence[str] | None = None,
    session=None,
) -> pd.DataFrame:
    """Share of enterprises using AI by country, NACE section and year (dataset ``isoc_eb_ain2``).

    Eurostat publishes this breakdown for enterprises with 10 or more persons
    employed only (``size_emp=GE10``) and for the NACE sections C, D, E, F, G,
    H, I, J, L, M and N (no financial sector K). Sub-section and aggregate codes
    (e.g. ``C10-C12``, ``L_M``) are dropped. Returns ``country``,
    ``nace_section``, ``year`` and ``eurostat_share`` (fraction).
    """
    filters: dict[str, Any] = {"indic_is": indic_is, "unit": unit, "size_emp": "GE10"}
    if countries:
        filters["geo"] = [ISO_TO_EUROSTAT.get(c.upper(), c.upper()) for c in countries]
    df = load_eurostat("isoc_eb_ain2", filters=None if path else filters, path=path, session=session)
    required = ("size_emp", "indic_is", "unit") if path is not None else ("size_emp",)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"industry benchmark lacks required dimensions: {missing}")
    for col in ("indic_is", "unit", "size_emp"):
        if col in df.columns:
            df = df[df[col] == filters[col]]
    if countries and "geo" in df.columns:
        df = df[df["geo"].isin(filters["geo"])]
    df = df[df["nace_r2"].astype(str).str.fullmatch(r"[A-U]")]
    _validate_eurostat_values(df)
    out = pd.DataFrame(
        {
            "country": df["geo"].map(lambda g: EUROSTAT_TO_ISO.get(g, g)),
            "nace_section": df["nace_r2"].astype(str),
            "year": pd.to_numeric(df["time"], errors="coerce").astype("Int64"),
            "eurostat_share": df["value"] / 100.0,
        }
    )
    return out[out["country"].str.len() == 2].reset_index(drop=True)


def load_benchmark(
    path: str | Path,
    value: str,
    keys: Sequence[str] = ("country", "year"),
    rename: Mapping[str, str] | None = None,
    percent: bool | None = None,
    value_name: str = "benchmark_share",
) -> pd.DataFrame:
    """Load any external benchmark from a CSV file (e.g. a national survey of AI use).

    ``rename`` maps the file's column names to AIDisclose keys (for example
    ``{"ref_year": "year", "naics_sector": "nace_section"}``; industry codes
    must already be mapped to the classification of the panel). Values are
    converted to fractions; ``percent=None`` treats them as percentages when
    a value contains a percent sign or any value exceeds 1. ISO country codes
    are upper-cased. Values at or below 1 without a percent sign are ambiguous;
    set ``percent=True`` for percentages expressed in that form.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if rename:
        df = df.rename(columns=dict(rename))
    missing = [c for c in [*keys, value] if c not in df.columns]
    if missing:
        raise ValueError(f"benchmark file lacks columns {missing}")
    out = df[list(keys)].copy()
    marked = df[value].str.contains("%", regex=False).any()
    if marked and percent is False:
        raise ValueError("percent=False conflicts with percent signs in benchmark values")
    raw = df[value].str.strip()
    missing_values = raw.str.lower().isin({"", "na", "n/a", "null", "nan", "."})
    vals = pd.to_numeric(raw.str.replace("%", "", regex=False).str.strip(), errors="coerce")
    invalid = ~missing_values & (vals.isna() | ~np.isfinite(vals))
    if invalid.any():
        raise ValueError(f"benchmark column {value!r} has {int(invalid.sum())} invalid or nonfinite value(s)")
    if percent is None:
        percent = bool(marked or (vals > 1).any())
    out[value_name] = vals / 100.0 if percent else vals
    if ((out[value_name] < 0) | (out[value_name] > 1)).any():
        raise ValueError("benchmark shares must be between 0 and 1 after conversion")
    if "country" in out.columns:
        out["country"] = out["country"].str.upper().map(lambda c: EUROSTAT_TO_ISO.get(c, c))
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out.dropna(subset=[value_name]).reset_index(drop=True)


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def correlation(x: Sequence[float], y: Sequence[float], method: str = "spearman") -> float:
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("x and y must be one-dimensional and have the same length")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("correlation inputs cannot contain infinite values")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if method == "spearman":
        a, b = _rank(a), _rank(b)
    return float(np.corrcoef(a, b)[0, 1])


def permutation_pvalue(
    x, y, method: str = "spearman", n_perm: int = 5000, seed: int = 0, blocks: Sequence[str] | None = None
) -> float:
    """Two-sided correlation permutation, optionally shuffling whole balanced blocks.

    With ``blocks`` (e.g. country identifiers in a country-year panel), the
    within-block row order is preserved and block trajectories are reassigned.
    Unequal block lengths are not exchangeable and return NaN.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    if not isinstance(n_perm, int) or isinstance(n_perm, bool) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("x and y must be one-dimensional and have the same length")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("correlation inputs cannot contain infinite values")
    if blocks is not None:
        if len(blocks) != len(a):
            raise ValueError("blocks must have the same length as x and y")
        if any(pd.isna(label) or not str(label).strip() for label in blocks):
            raise ValueError("blocks cannot contain missing identifiers")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    block_values = None if blocks is None else np.asarray(blocks, dtype=object)[mask]
    obs = correlation(a, b, method)
    if np.isnan(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    exact = False
    if block_values is not None:
        labels = list(dict.fromkeys(block_values.tolist()))
        indices = [np.flatnonzero(block_values == label) for label in labels]
        if len(labels) < 2 or len({len(ix) for ix in indices}) != 1:
            return float("nan")
        trajectories = [b[ix] for ix in indices]

        def permuted(p):
            shuffled = np.empty_like(b)
            for ix, j in zip(indices, p):
                shuffled[ix] = trajectories[j]
            return shuffled

        if len(labels) <= 7:
            exact = True
            perms = itertools.permutations(range(len(labels)))
            vals = [correlation(a, permuted(p), method) for p in perms]
        else:
            vals = []
            for _ in range(n_perm):
                p = rng.permutation(len(labels))
                vals.append(correlation(a, permuted(p), method))
    elif len(a) <= 7:  # exact enumeration
        exact = True
        perms = itertools.permutations(range(len(b)))
        vals = [correlation(a, b[list(p)], method) for p in perms]
    else:
        vals = [correlation(a, rng.permutation(b), method) for _ in range(n_perm)]
    vals = np.asarray(vals)
    if len(vals) == 0:
        raise ValueError("n_perm must be positive")
    extreme = int(np.sum(np.abs(vals) >= abs(obs) - 1e-12))
    return float(extreme / len(vals) if exact else (extreme + 1) / (len(vals) + 1))


def compare_with_external(
    group: pd.DataFrame,
    external: pd.DataFrame,
    measures: Sequence[str] = ("share_keyword", "share_claim", "share_substantive"),
    group_keys: Sequence[str] = ("country", "fiscal_year"),
    external_keys: Sequence[str] = ("country", "year"),
    external_value: str = "eurostat_share",
    year_lag: int = 0,
    min_firms: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join group-level indicators with an external benchmark and correlate them.

    ``year_lag`` shifts the disclosure year before matching (e.g. ``1`` compares
    fiscal year *t* reports with survey year *t+1*).
    Returns ``(merged, summary)`` where ``summary`` has one row per measure with
    Spearman and Pearson correlations, a permutation p-value and the mean
    absolute difference in percentage points.
    """
    if not group_keys or len(group_keys) != len(external_keys):
        raise ValueError("group_keys and external_keys must have the same nonzero length")
    if len(set(group_keys)) != len(group_keys) or len(set(external_keys)) != len(external_keys):
        raise ValueError("comparison keys must be unique")
    g = group.copy()
    if "n_firms" in g.columns:
        g = g[g["n_firms"] >= min_firms]
    g = g.rename(columns=dict(zip(group_keys, external_keys)))
    # The public API and CLI accept keys in any order; the year need not be
    # the last key (e.g. country, year, industry).
    if "fiscal_year" in group_keys:
        year_col = external_keys[list(group_keys).index("fiscal_year")]
    elif "year" in external_keys:
        year_col = "year"
    else:
        year_col = external_keys[-1]  # legacy custom year-key convention
    g[year_col] = pd.to_numeric(g[year_col], errors="coerce").astype("Int64") + year_lag
    ext = external.copy()
    ext[year_col] = pd.to_numeric(ext[year_col], errors="coerce").astype("Int64")
    for name, frame in (("group", g), ("external", ext)):
        keys_frame = frame[list(external_keys)]
        missing = keys_frame.isna() | keys_frame.astype(str).apply(lambda s: s.str.strip().eq(""))
        if missing.any().any():
            raise ValueError(f"{name} comparison keys contain missing values")
        if frame.duplicated(list(external_keys)).any():
            raise ValueError(f"{name} comparison keys must be unique")
    # Validate all available input observations, including unmatched rows, so
    # malformed values cannot silently change the population under comparison.
    for frame, name, cols in (
        (g, "group", [m for m in measures if m in g.columns]),
        (ext, "external", [external_value]),
    ):
        for col in cols:
            if col not in frame:
                raise ValueError(f"{name} comparison lacks column {col!r}")
            original = frame[col]
            numeric = pd.to_numeric(original, errors="coerce")
            nonblank = original.notna() & original.astype(str).str.strip().ne("")
            if (nonblank & numeric.isna()).any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"{name} comparison column {col!r} must contain finite numbers or missing values")
            frame[col] = numeric
    merged = g.merge(ext, on=list(external_keys), how="inner", validate="one_to_one")
    rows = []
    for m in measures:
        if m not in merged.columns:
            continue
        valid = merged[[m, external_value]].notna().all(axis=1)
        block_col = "country" if "country" in external_keys and "country" in merged else None
        panel_values = merged.loc[valid].copy()
        within_cols = [c for c in external_keys if c != block_col] if block_col else []
        if block_col:
            panel_values = panel_values.sort_values([block_col, *within_cols])
            trajectories = [
                tuple(tuple(row) for row in sub[within_cols].itertuples(index=False, name=None))
                for _, sub in panel_values.groupby(block_col, sort=False)
            ]
            balanced = bool(trajectories) and all(keys == trajectories[0] for keys in trajectories[1:])
            blocks = panel_values[block_col].astype(str).tolist()
        else:
            balanced = True
            blocks = None
        p_perm = (
            permutation_pvalue(
                panel_values[m],
                panel_values[external_value],
                "spearman",
                blocks=blocks,
            )
            if balanced
            else float("nan")
        )
        if block_col:
            if not balanced:
                p_note = "not estimable: unbalanced country panel"
            elif np.isfinite(p_perm):
                p_note = "country-block (balanced trajectories)"
            else:
                p_note = "not estimable: too few countries or undefined correlation"
        else:
            p_note = "row permutation (no country blocks)"
        rows.append(
            {
                "measure": m,
                "n_pairs": int(merged[[m, external_value]].dropna().shape[0]),
                "spearman": correlation(merged[m], merged[external_value], "spearman"),
                "pearson": correlation(merged[m], merged[external_value], "pearson"),
                "p_perm_spearman": p_perm,
                "permutation_method": p_note,
                "mean_abs_diff_pp": float(np.nanmean(np.abs(merged[m] - merged[external_value])) * 100)
                if valid.any()
                else float("nan"),
            }
        )
    return merged, pd.DataFrame(rows)
