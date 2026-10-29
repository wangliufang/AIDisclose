"""Stratified sampling of candidate sentences for human annotation."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

ANNOTATION_COLUMNS = ["gold_state", "gold_reason", "annotator", "notes"]

_KEY_FIELDS = (
    "sentence_id",
    "text",
    "filing_id",
    "firm_id",
    "sampling_weight",
    "sampling_stratum",
    "recall_weight",
    "recall_stratum",
    "dataset_sha256",
    "coverage_verified",
    "classification_complete",
)
_CONTEXT_FIELDS = ("firm_name", "fiscal_year", "section", "prev_text", "next_text")
_KEY_FIELDS_V2 = _KEY_FIELDS + _CONTEXT_FIELDS + ("sample_schema",)


def _key_digest(key: pd.DataFrame) -> str:
    # Round-trip through CSV to make numerical spelling identical at creation
    # and later loading, and make the digest insensitive to row ordering.
    normalized = pd.read_csv(io.StringIO(key.to_csv(index=False)), dtype=str, keep_default_na=False)
    if "sample_schema" in normalized:
        if not normalized["sample_schema"].eq("2").all():
            raise ValueError("unsupported or inconsistent sample key schema")
        key_fields = _KEY_FIELDS_V2
    else:
        key_fields = _KEY_FIELDS  # legacy seal without context protection
    fields = [c for c in key_fields if c in normalized]
    if "sentence_id" not in fields:
        raise ValueError("sample key requires sentence_id")
    rows = normalized[fields].sort_values("sentence_id").to_dict(orient="records")
    payload = json.dumps({"fields": fields, "rows": rows}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seal_sample_key(key: pd.DataFrame) -> pd.DataFrame:
    if key.empty or key["sentence_id"].duplicated().any():
        raise ValueError("sample key must be nonempty with unique identifiers")
    out = key.copy()
    out["sample_schema"] = "2"
    out["sample_expected_n"] = str(len(out))
    out["sample_sha256"] = _key_digest(out)
    return out


def verify_sample_key(key: pd.DataFrame) -> None:
    if key.empty or "sample_expected_n" not in key or "sample_sha256" not in key:
        raise ValueError("sample key lacks a completeness seal; regenerate the sample")
    if not key["sample_expected_n"].astype(str).eq(str(len(key))).all():
        raise ValueError("sample key is incomplete or has duplicate rows")
    if not key["sample_sha256"].astype(str).eq(_key_digest(key)).all():
        raise ValueError("sample key contents differ from the generated sample")
    if key["sentence_id"].duplicated().any():
        raise ValueError("sample key contains duplicate identifiers")


def sample_for_annotation(
    sentences: pd.DataFrame,
    n: int = 400,
    strata: Iterable[str] = ("state",),
    min_per_stratum: int = 20,
    seed: int = 42,
    include_false_positives: bool = True,
) -> pd.DataFrame:
    """Draw a stratified random sample and attach sampling weights.

    Allocation is proportional to stratum size with a floor of
    ``min_per_stratum`` (so that rare states such as SUBSTANTIVE are well
    represented). Each sampled row receives ``sampling_weight`` = stratum
    size / sampled rows in the stratum, which lets
    :func:`~aidisclose.validate.metrics.classification_report` estimate
    population-level accuracy.
    """
    df = sentences.copy()
    if "classification_status" in df.columns:
        df = df[df["classification_status"] == "ok"]
    if not include_false_positives:
        df = df[df["reason"] != "FALSE_POSITIVE"]
    df = df[df["raw_terms"].astype(str) != ""]
    strata = list(strata)
    if df.empty:
        raise ValueError("no candidate sentences to sample from")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(min_per_stratum, int) or isinstance(min_per_stratum, bool) or min_per_stratum < 0:
        raise ValueError("min_per_stratum must be a non-negative integer")
    if not strata:
        raise ValueError("at least one stratification column is required")
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(strata, dropna=False, sort=True)}
    target = min(n, len(df))
    if target < len(groups):
        raise ValueError("n must be at least the number of nonempty strata to estimate population metrics")
    # Guarantee positive inclusion in every stratum; fill the remaining budget
    # by greatest shortfall from proportional allocation, respecting capacities.
    floor = max(1, min_per_stratum)
    desired = {k: min(len(g), floor) for k, g in groups.items()}
    alloc = {k: 1 for k in groups}
    while sum(alloc.values()) < target:
        eligible = [k for k, g in groups.items() if alloc[k] < len(g)]
        floor_eligible = [k for k in eligible if alloc[k] < desired[k]]
        candidates = floor_eligible or eligible
        k = max(candidates, key=lambda key: target * len(groups[key]) / len(df) - alloc[key])
        alloc[k] += 1
    parts = []
    for k, g in groups.items():
        m = alloc[k]
        if m <= 0:
            continue
        ix = rng.choice(len(g), size=m, replace=False)
        part = g.iloc[np.sort(ix)].copy()
        key_values = k if isinstance(k, tuple) else (k,)
        # Typed JSON avoids collisions caused by delimiters, numeric strings,
        # or missing values in different combinations of grouping columns.
        part["sampling_stratum"] = json.dumps(
            [
                {"type": "missing" if pd.isna(x) else type(x).__name__, "value": "" if pd.isna(x) else str(x)}
                for x in key_values
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        part["sampling_weight"] = len(g) / m
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle so strata are not visible
    for col in ANNOTATION_COLUMNS:
        out[col] = ""
    return out


def blind(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a sample into an annotator file (no predictions) and a key file."""
    visible = ["sentence_id", "firm_name", "fiscal_year", "section", "prev_text", "text", "next_text"]
    visible = [c for c in visible if c in sample.columns] + ANNOTATION_COLUMNS
    hidden = [c for c in sample.columns if c not in ANNOTATION_COLUMNS]
    return sample[visible].copy(), sample[hidden].copy()


def merge_sample_key(gold: pd.DataFrame, key: pd.DataFrame, on: str = "sentence_id") -> pd.DataFrame:
    """Join a complete annotation sample to its key, rejecting loss, duplication and changed text."""
    for frame, name in ((gold, "gold"), (key, "key")):
        if on not in frame or frame[on].isna().any() or (frame[on].astype(str).str.strip() == "").any():
            raise ValueError(f"{name} needs nonempty {on} identifiers")
        if frame[on].duplicated().any():
            raise ValueError(f"{name} contains duplicate {on} identifiers")
    if set(gold[on].astype(str)) != set(key[on].astype(str)):
        raise ValueError("annotation and key identifiers do not match; sample must be complete")
    for field in ("text", *_CONTEXT_FIELDS):
        if field in gold and field in key:
            checked = gold[[on, field]].merge(key[[on, field]], on=on, validate="one_to_one")
            left = checked[field + "_x"].fillna("").astype(str).str.strip()
            right = checked[field + "_y"].fillna("").astype(str).str.strip()
            if (left != right).any():
                raise ValueError(f"annotation {field} differs from the sampled {field}")
    protected = {"sampling_weight", "sampling_stratum", "recall_weight", "recall_stratum", "filing_id", "firm_id"}
    gold = gold.drop(columns=[c for c in protected if c in gold and c in key])
    cols = [c for c in key if c != on and c not in gold]
    return gold.merge(key[[on, *cols]], on=on, how="left", validate="one_to_one")


def merge_annotations(gold: pd.DataFrame, predictions: pd.DataFrame, on: str = "sentence_id") -> pd.DataFrame:
    """Join annotated rows with predictions; rows without a gold label are dropped."""
    g = gold.copy()
    collisions = (
        {"state", "reason", "classification_status", "confidence", "backend", "details", "_pred_text"}
        & set(g.columns)
        & set(predictions.columns)
    )
    if collisions:
        raise ValueError(f"gold data contains prediction columns: {sorted(collisions)}")
    g["gold_state"] = g["gold_state"].astype(str).str.strip().str.upper().str.replace(" ", "_")
    g = g[g["gold_state"].isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"])]
    pred_cols = [c for c in predictions.columns if c not in g.columns or c == on]
    return g.merge(predictions[pred_cols], on=on, how="inner", validate="one_to_one")
