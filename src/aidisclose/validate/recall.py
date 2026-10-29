"""Recall audit: how many AI claims does the lexicon miss?

Precision-oriented validation (``aidisclose sample`` / ``evaluate``) only sees
sentences that the lexicon retrieved. To estimate what it *misses*, the
pipeline can keep a *recall pool* (``Pipeline(recall_random=..,
recall_probe_cap=..)``): per document, a random sample of the sentences that
the measurement lexicon did not match (including baseline-only hits), stratified into

* ``probe``  sentences that contain an AI-adjacent probe term ("algorithm",
  "automation", "predictive", ...), where missed claims are more likely, and
* ``random`` all other non-retrieved sentences,

with their within-document inclusion probabilities. :func:`sample_recall`
draws an annotation sample from the pool and :func:`estimate_recall` turns the
annotations into Horvitz-Thompson estimates of the number of missed
SUBSTANTIVE (and WEAK) sentences, and, combined with the number of retrieved
true SUBSTANTIVE sentences, of the lexicon's recall. Confidence intervals come
from a stratified bootstrap over both annotated samples.

Scope: the estimates refer to the processed sections of the processed
(English) documents only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .sampling import ANNOTATION_COLUMNS, merge_sample_key

CLAIM_STATES = ("SUBSTANTIVE", "WEAK")


def validate_recall_coverage(results) -> None:
    """Require first-stage representation of every nonempty document stratum."""
    config = results.provenance.get("recall_pool")
    if not isinstance(config, dict):
        raise ValueError("recall pool sampling configuration is missing; coverage cannot be verified")
    docs = results.documents
    for col in ("filing_id", "status", "n_probe_sentences", "n_other_sentences"):
        if col not in docs:
            raise ValueError(f"recall coverage requires document field {col}")
    if docs["filing_id"].duplicated().any():
        raise ValueError("duplicate filings prevent recall coverage verification")
    pool = results.recall_pool
    ok_ids = set(docs.loc[docs["status"] == "ok", "filing_id"])
    if not ok_ids:
        raise ValueError("recall coverage requires at least one processed document")
    if not pool.empty and (
        pool["sentence_id"].duplicated().any()
        or not set(pool["filing_id"]).issubset(ok_ids)
        or not set(pool["pool"]).issubset({"probe", "random"})
    ):
        raise ValueError("recall pool contains duplicate, orphan or unknown-stratum rows")
    for row in docs[docs["status"] == "ok"].itertuples():
        for name, count, cap in (
            ("probe", row.n_probe_sentences, config.get("probe_cap_per_document", 0)),
            ("random", row.n_other_sentences, config.get("random_per_document", 0)),
        ):
            count = int(count)
            part = pool[(pool["filing_id"] == row.filing_id) & (pool["pool"] == name)]
            expected = min(count, int(cap))
            if count and expected <= 0:
                raise ValueError(f"recall stratum {name} in {row.filing_id} has zero inclusion probability")
            if len(part) != expected or (
                len(part)
                and (
                    part["sentence_id"].duplicated().any()
                    or not (pd.to_numeric(part["stratum_size"]) == count).all()
                    or not (pd.to_numeric(part["n_drawn"]) == expected).all()
                    or not np.allclose(pd.to_numeric(part["inclusion_prob"]), expected / count)
                )
            ):
                raise ValueError(f"recall pool incomplete or inconsistent for {row.filing_id} / {name}")


def sample_recall(
    pool: pd.DataFrame, n: int = 300, share_probe: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw ``n`` pool sentences for annotation; returns ``(to_annotate, key)``.

    ``share_probe`` of the sample comes from the probe stratum (or all of it if
    it is smaller). The key file carries ``recall_weight``, the inverse of the
    overall inclusion probability (document stage x sample stage).
    """
    if pool.empty:
        raise ValueError("the recall pool is empty; run the pipeline with recall_random/recall_probe_cap > 0")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 <= share_probe <= 1:
        raise ValueError("share_probe must be between 0 and 1")
    if not set(pool["pool"]).issubset({"probe", "random"}):
        raise ValueError("pool contains unknown strata")
    probabilities = pd.to_numeric(pool["inclusion_prob"], errors="coerce")
    if (
        probabilities.isna().any()
        or (~np.isfinite(probabilities)).any()
        or ((probabilities <= 0) | (probabilities > 1)).any()
    ):
        raise ValueError("inclusion_prob must be finite and in (0, 1]")
    rng = np.random.default_rng(seed)
    parts = []
    strata = {name: g for name, g in pool.groupby("pool")}
    want = {"probe": int(round(n * share_probe))}
    want["random"] = n - want["probe"]
    # move unused allocation to the other stratum
    for a, b in (("probe", "random"), ("random", "probe")):
        size = len(strata.get(a, []))
        if want.get(a, 0) > size:
            want[b] = want.get(b, 0) + want[a] - size
            want[a] = size
    if any(want.get(name, 0) <= 0 for name in strata):
        raise ValueError("every nonempty recall stratum must receive at least one annotation")
    for name, g in strata.items():
        m = min(len(g), want.get(name, 0))
        if m <= 0:
            continue
        ix = np.sort(rng.choice(len(g), size=m, replace=False))
        part = g.iloc[ix].copy()
        p_doc = pd.to_numeric(part["inclusion_prob"], errors="coerce").fillna(1.0)
        part["recall_weight"] = (1.0 / p_doc) * (len(g) / m)
        part["recall_stratum"] = name
        part["recall_stratum_pool_size"] = len(g)
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for col in ANNOTATION_COLUMNS:
        sample[col] = ""
    visible = ["sentence_id", "firm_name", "fiscal_year", "section", "prev_text", "text", "next_text"]
    visible = [c for c in visible if c in sample.columns] + ANNOTATION_COLUMNS
    hidden = [c for c in sample.columns if c not in ANNOTATION_COLUMNS]
    return sample[visible].copy(), sample[hidden].copy()


def _norm_state(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().str.replace(" ", "_")


def _ht_totals(df: pd.DataFrame, weight: str) -> dict[str, float]:
    w = pd.to_numeric(df[weight], errors="coerce").fillna(0.0).to_numpy()
    st = _norm_state(df["gold_state"]).to_numpy()
    return {
        "substantive": float(np.sum(w * (st == "SUBSTANTIVE"))),
        "claim": float(np.sum(w * np.isin(st, CLAIM_STATES))),
        "population": float(np.sum(w)),
    }


def estimate_recall(
    recall_gold: pd.DataFrame,
    main_gold: pd.DataFrame | None = None,
    found_substantive: float | None = None,
    found_claim: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    found_source: str | None = None,
) -> dict[str, Any]:
    """Estimate missed claims and lexicon recall.

    Parameters
    ----------
    recall_gold:
        Annotated recall sample merged with its key (needs ``gold_state``,
        ``recall_weight`` and ``recall_stratum``).
    main_gold:
        Annotated precision sample merged with its key (``gold_state``,
        ``sampling_weight``, optional ``sampling_stratum``), drawn from *all*
        retrieved sentences. It estimates how many retrieved sentences are
        truly SUBSTANTIVE (or claims).
    found_substantive, found_claim:
        Alternatively, the numbers of retrieved true SUBSTANTIVE / claim
        sentences (e.g. predicted counts, which assumes the classifier's
        errors cancel; the report says which source was used).
    found_source:
        Description of supplied counts; use an explicit prediction label when
        they are machine-generated. Not used with a main annotated sample.
    """
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rg = recall_gold.copy()
    if not _norm_state(rg["gold_state"]).isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"]).all():
        raise ValueError("all sampled recall rows must have valid gold labels")
    if rg.empty:
        raise ValueError("no annotated rows in the recall sample")
    if "recall_weight" not in rg.columns:
        raise ValueError("the recall sample needs 'recall_weight' (merge it with the key file)")
    if (
        "recall_stratum" not in rg
        or rg["recall_stratum"].isna().any()
        or rg["recall_stratum"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("recall_stratum must be present and nonempty")
    mg = None
    if main_gold is not None:
        mg = main_gold.copy()
        if not _norm_state(mg["gold_state"]).isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"]).all():
            raise ValueError("all sampled main rows must have valid gold labels")
        if "sampling_weight" not in mg.columns:
            raise ValueError("the main sample needs 'sampling_weight' (merge it with the key file)")
    elif found_substantive is None:
        raise ValueError("give main_gold or found_substantive")

    for frame, column in ((rg, "recall_weight"), (mg, "sampling_weight")):
        if frame is not None:
            if frame.empty:
                raise ValueError("annotated samples must not be empty")
            w = pd.to_numeric(frame[column], errors="coerce")
            if w.isna().any() or (~np.isfinite(w)).any() or (w <= 0).any():
                raise ValueError(f"{column} must contain finite positive weights")
    for value in (found_substantive, found_claim):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("supplied counts must be finite and non-negative")
    if main_gold is None and found_claim is not None and found_claim < found_substantive:
        raise ValueError("found_claim cannot be smaller than found_substantive")
    if found_source is not None and not str(found_source).strip():
        raise ValueError("found_source cannot be blank")
    if main_gold is not None and found_source is not None:
        raise ValueError("found_source applies only to supplied counts")

    def point(r: pd.DataFrame, m: pd.DataFrame | None) -> dict[str, float]:
        missed = _ht_totals(r, "recall_weight")
        if m is not None:
            found = _ht_totals(m, "sampling_weight")
            f_sub, f_claim = found["substantive"], found["claim"]
        else:
            f_sub = float(found_substantive)
            f_claim = float(found_claim) if found_claim is not None else float("nan")
        out = {
            "missed_substantive": missed["substantive"],
            "missed_claim": missed["claim"],
            "non_retrieved_sentences": missed["population"],
            "found_substantive": f_sub,
            "found_claim": f_claim,
        }
        out["recall_substantive"] = (
            f_sub / (f_sub + missed["substantive"]) if f_sub + missed["substantive"] else float("nan")
        )
        out["recall_claim"] = (
            f_claim / (f_claim + missed["claim"]) if f_claim == f_claim and f_claim + missed["claim"] else float("nan")
        )
        out["missed_substantive_rate"] = (
            missed["substantive"] / missed["population"] if missed["population"] else float("nan")
        )
        return out

    est = point(rg, mg)
    rng = np.random.default_rng(seed)

    def resample(df: pd.DataFrame, stratum: str | None) -> pd.DataFrame:
        if stratum and stratum in df.columns:
            parts = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in df.groupby(stratum)]
            return pd.concat(parts)
        return df.iloc[rng.integers(0, len(df), len(df))]

    boots: dict[str, list[float]] = {k: [] for k in est}
    insufficient = bool((rg.groupby("recall_stratum", dropna=False).size() < 2).any())
    if mg is not None:
        if "sampling_stratum" in mg:
            if mg["sampling_stratum"].isna().any() or mg["sampling_stratum"].astype(str).str.strip().eq("").any():
                raise ValueError("sampling_stratum cannot contain missing identifiers")
            insufficient = insufficient or bool((mg.groupby("sampling_stratum", dropna=False).size() < 2).any())
        else:
            insufficient = insufficient or len(mg) < 2
    for _ in range(0 if insufficient else n_boot):
        b = point(resample(rg, "recall_stratum"), resample(mg, "sampling_stratum") if mg is not None else None)
        for k, v in b.items():
            boots[k].append(v)
    ci = {}
    for k, vals in boots.items():
        arr = np.asarray([v for v in vals if v == v], dtype=float)
        ci[k] = (
            [float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))] if len(arr) else [None, None]
        )
    by_stratum = (
        rg.assign(_sub=_norm_state(rg["gold_state"]) == "SUBSTANTIVE")
        .groupby("recall_stratum")
        .agg(n_annotated=("_sub", "size"), n_substantive=("_sub", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "estimates": est,
        "ci": ci,
        "confidence_level": 1 - alpha,
        "n_recall_annotated": int(len(rg)),
        "n_main_annotated": int(len(mg)) if mg is not None else 0,
        "found_source": "main annotated sample" if mg is not None else (found_source or "supplied counts"),
        "by_stratum": [
            {k: (int(v) if isinstance(v, (np.integer, bool, np.bool_)) else v) for k, v in r.items()}
            for r in by_stratum
        ],
        "n_boot": n_boot,
        "interval_status": "not estimable: fewer than two annotations in at least one stratum"
        if insufficient
        else "approximate",
        "interval_method": "approximate row-stratified bootstrap; does not fully reproduce the two-stage document design",
    }


def recall_to_markdown(rep: dict[str, Any], title: str = "Recall audit") -> str:
    def f(x):
        return "n/a" if x is None or x != x else (f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}")

    e, ci = rep["estimates"], rep["ci"]
    rows = [
        ("Recall of SUBSTANTIVE sentences", "recall_substantive"),
        ("Recall of claims (SUBSTANTIVE or WEAK)", "recall_claim"),
        ("Missed SUBSTANTIVE sentences (estimated total)", "missed_substantive"),
        ("Missed claims (estimated total)", "missed_claim"),
        ("Retrieved SUBSTANTIVE sentences (estimated)", "found_substantive"),
        ("Share of non-retrieved sentences that are SUBSTANTIVE", "missed_substantive_rate"),
    ]
    lines = [
        f"# {title}",
        "",
        f"Annotated recall sample: {rep['n_recall_annotated']} sentences; retrieved-sentence totals from: "
        f"{rep['found_source']} ({rep['n_main_annotated']} annotated). "
        f"{int(rep['confidence_level'] * 100)}% stratified bootstrap intervals ({rep['n_boot']} replicates).",
        "These intervals are approximate and do not fully reproduce the two-stage document sampling design.",
        f"Interval status: {rep.get('interval_status', 'approximate')}.",
        "",
        "| Quantity | Estimate | CI |",
        "|---|---:|---|",
    ]
    for label, key in rows:
        lo, hi = ci.get(key, [None, None])
        lines.append(f"| {label} | {f(e.get(key))} | [{f(lo)}, {f(hi)}] |")
    lines += ["", "| Stratum | Annotated | SUBSTANTIVE |", "|---|---:|---:|"]
    for r in rep["by_stratum"]:
        lines.append(f"| {r['recall_stratum']} | {r['n_annotated']} | {r['n_substantive']} |")
    return "\n".join(lines) + "\n"


def merge_recall_key(gold: pd.DataFrame, key: pd.DataFrame, on: str = "sentence_id") -> pd.DataFrame:
    return merge_sample_key(gold, key, on=on)


def found_from_predictions(sentences: pd.DataFrame, states: Sequence[str] = ("SUBSTANTIVE",)) -> float:
    """Count valid predictions retrieved by the main lexicon, not baseline-only rows."""
    mask = sentences["state"].isin(list(states))
    if "classification_status" in sentences:
        mask &= sentences["classification_status"] == "ok"
    if "raw_terms" in sentences:
        mask &= sentences["raw_terms"].fillna("").astype(str).str.strip().ne("")
    return float(mask.sum())
