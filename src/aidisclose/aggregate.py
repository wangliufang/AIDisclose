"""Aggregate sentence labels into firm-year and group-year AI adoption indicators.

Firm-year indicators
--------------------
``keyword_any``        the report contains a keyword-baseline hit (plain keyword measure)
``claim_any``          at least one WEAK or SUBSTANTIVE focal-firm claim
``substantive_any``    at least one SUBSTANTIVE claim
``n_*``                sentence counts by state
``*_per_10k_words``    sentence counts scaled by report length
``substantive_share``  SUBSTANTIVE sentences / sentences with any lexicon hit
``evidence``           identifiers of up to ``n_evidence`` (default 3) SUBSTANTIVE sentences,
                       highest confidence first; ``Results.evidence()`` returns all of them
``n_classifier_errors`` sentences the backend could not label (e.g. unparseable model replies)
``classification_complete`` whether all retrieved candidate sentences received usable labels

By technique and by type of claim
----------------------------------------
``n_substantive_<category>``  SUBSTANTIVE sentences with a term of that lexicon category
                              (general, ml, genai, nlp, cv, automation); a sentence can count
                              for several categories
``substantive_<category>_any`` indicator of the above
``n_substantive_<reason>``    SUBSTANTIVE sentences by reason (deployment, product, investment)
``n_weak_<reason>``           WEAK sentences by reason (forward_looking, hedged,
                              risk_boilerplate, vague)

Dynamics
---------------
``n_substantive_new``  SUBSTANTIVE sentences that do not repeat a lexicon-hit sentence of the
                       firm's report for the previous fiscal year (token-set Jaccard similarity
                       below ``repeat_threshold``, digits ignored, so a claim restated with
                       updated figures counts as repeated)
``substantive_novelty`` ``n_substantive_new / n_substantive``; empty when the previous fiscal
                       year is not in the panel or there is no SUBSTANTIVE sentence
``first_observed_year`` first fiscal year of the firm in the panel
``first_substantive_year`` first fiscal year with ``substantive_any``
``substantive_onset``  1 in the firm's first SUBSTANTIVE year if the previous fiscal year was
                       observed (without a SUBSTANTIVE claim), empty in that year if the
                       previous year was not observed (the onset is censored), 0 in all
                       other years

Group-year indicators are firm-weighted means (or weighted by a user column,
e.g. an employment figure merged with ``firm_attributes``), plus
``gap_keyword_substantive_pp``: keyword minus substantive prevalence in
percentage points only when all positive-weight firms have observable substantive indicators.
Observed-only shares and worst-case bounds are reported separately when outcomes are missing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .industry import sic_to_nace_section

CATEGORIES = ("general", "ml", "genai", "nlp", "cv", "automation")
SUBSTANTIVE_REASONS = ("DEPLOYMENT", "PRODUCT", "INVESTMENT")
WEAK_REASONS = ("FORWARD_LOOKING", "HEDGED", "RISK_BOILERPLATE", "VAGUE")
_WORD = re.compile(r"[a-z]+|\d+")


def _nace_from_row(row: pd.Series, industry_map: Mapping[str, str] | Callable | None) -> str | None:
    code = str(row.get("industry_code") or "")
    scheme = str(row.get("industry_scheme") or "")
    if industry_map is not None:
        if callable(industry_map):
            return industry_map(row)
        if code in industry_map:
            return industry_map[code]
    extra = row.get("extra")
    if isinstance(extra, str) and extra.strip().startswith("{"):
        try:
            nace = json.loads(extra).get("nace_section")
            if nace:
                return nace
        except json.JSONDecodeError:
            pass
    if scheme == "SIC":
        return sic_to_nace_section(code)
    if scheme == "NACE" and code:
        return code[:1].upper()
    return None


def _tokens(text: str) -> frozenset[str]:
    return frozenset("0" if t.isdigit() else t for t in _WORD.findall(str(text).lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def firm_year_panel(
    results,
    industry_map: Mapping[str, str] | Callable | None = None,
    n_evidence: int = 3,
    firm_attributes: pd.DataFrame | None = None,
    repeat_threshold: float = 0.8,
    unknown_years: str = "error",
    xbrl_periods: str = "error",
) -> pd.DataFrame:
    """Build a firm-year panel from :class:`~aidisclose.pipeline.Results`.

    When a firm has several processed reports for the same fiscal year (e.g. a
    10-K and a 10-K/A), the longest report is kept.

    ``firm_attributes`` is merged on ``firm_id`` (and ``fiscal_year`` if it has
    that column), e.g. to add employment for weighted group means.
    Unknown fiscal years raise an error unless ``unknown_years="exclude"``;
    exclusion emits a warning with the number of reports removed.
    XBRL indexes can contain interim reports; users must verify that every
    included XBRL report is annual before passing ``xbrl_periods="reviewed"``.
    Multiple XBRL period ends for the same firm-year are rejected even after
    confirmation; select one verified report per firm-year before processing.
    """
    if unknown_years not in {"error", "exclude"}:
        raise ValueError("unknown_years must be 'error' or 'exclude'")
    if xbrl_periods not in {"error", "reviewed"}:
        raise ValueError("xbrl_periods must be 'error' or 'reviewed'")
    docs = results.documents.copy()
    docs = docs[docs["status"] == "ok"].copy()
    if docs.empty:
        return pd.DataFrame()
    docs["fiscal_year"] = pd.to_numeric(docs["fiscal_year"], errors="coerce").astype("Int64")
    missing_year = docs["fiscal_year"].isna()
    if missing_year.any():
        count = int(missing_year.sum())
        if unknown_years == "error":
            raise ValueError(
                f"{count} processed report(s) have an unknown fiscal year; correct metadata or explicitly exclude them"
            )
        import warnings

        warnings.warn(f"excluding {count} processed report(s) with unknown fiscal year", UserWarning, stacklevel=2)
        docs = docs.loc[~missing_year].copy()
        if docs.empty:
            raise ValueError("no processed reports with known fiscal years remain after exclusion")
    xbrl = docs[docs["source"].astype(str).str.lower().eq("xbrl")].copy()
    if not xbrl.empty:
        if xbrl_periods != "reviewed":
            raise ValueError("XBRL report periods are unverified: select annual reports and explicitly confirm review")
        periods = pd.to_datetime(xbrl["period_end"], errors="coerce", format="%Y-%m-%d")
        if periods.isna().any():
            raise ValueError("XBRL reports need valid period_end values before annual aggregation")
        xbrl["_period_checked"] = periods
        if (xbrl.groupby(["firm_id", "fiscal_year"], dropna=False)["_period_checked"].nunique() > 1).any():
            raise ValueError("multiple XBRL period ends in a firm-year: select one verified annual report")
    docs = docs.sort_values(["firm_id", "fiscal_year", "n_words"], ascending=[True, True, False])
    docs = docs.drop_duplicates(["firm_id", "fiscal_year"], keep="first")
    docs["nace_section"] = docs.apply(lambda r: _nace_from_row(r, industry_map), axis=1)

    sent = results.sentences
    if not sent.empty:
        sent = sent[sent["filing_id"].isin(set(docs["filing_id"]))]
    evidence: dict[str, list[str]] = {}
    if not sent.empty:
        subs = sent[sent["state"] == "SUBSTANTIVE"].copy()
        subs["confidence"] = pd.to_numeric(subs["confidence"], errors="coerce")
        subs = subs.sort_values(["filing_id", "confidence"], ascending=[True, False])
        for fid, grp in subs.groupby("filing_id", sort=False):
            evidence[fid] = grp["sentence_id"].head(n_evidence).tolist()

    words = docs["n_words"].astype(float).replace(0, np.nan)
    panel = pd.DataFrame(
        {
            "firm_id": docs["firm_id"],
            "firm_name": docs["firm_name"],
            "fiscal_year": docs["fiscal_year"],
            "period_end": docs["period_end"],
            "country": docs["country"],
            "industry_code": docs["industry_code"],
            "industry_scheme": docs["industry_scheme"],
            "nace_section": docs["nace_section"],
            "source": docs["source"],
            "form": docs["form"] if "form" in docs else "",
            "filing_id": docs["filing_id"],
            "n_words": docs["n_words"].astype(int),
            "n_baseline_sentences": docs["n_baseline_sentences"].astype(int),
            "n_keyword_sentences": docs["n_keyword_sentences"].astype(int),
            "n_substantive": docs["n_substantive"].astype(int),
            "n_weak": docs["n_weak"].astype(int),
            "n_no_claim": docs["n_no_claim"].astype(int),
            "n_false_positive": docs["n_false_positive"].astype(int),
            "n_classifier_errors": pd.to_numeric(docs.get("n_classifier_errors", 0), errors="coerce")
            .fillna(0)
            .astype(int),
        }
    ).reset_index(drop=True)
    panel["keyword_any"] = (panel["n_baseline_sentences"] > 0).astype(int)
    panel["classification_complete"] = panel["n_classifier_errors"] == 0
    n_candidates = (
        pd.to_numeric(docs.get("n_candidates", pd.Series(0, index=docs.index)), errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    panel["n_classified_sentences"] = (n_candidates - panel["n_classifier_errors"]).clip(lower=0)
    panel["claim_any"] = pd.array(
        [
            1 if sub + weak > 0 else (0 if complete else pd.NA)
            for sub, weak, complete in zip(panel["n_substantive"], panel["n_weak"], panel["classification_complete"])
        ],
        dtype="Int64",
    )
    panel["substantive_any"] = pd.array(
        [
            1 if count > 0 else (0 if complete else pd.NA)
            for count, complete in zip(panel["n_substantive"], panel["classification_complete"])
        ],
        dtype="Int64",
    )
    words = panel["n_words"].astype(float).replace(0, np.nan)
    panel["keyword_per_10k_words"] = (panel["n_baseline_sentences"] / words * 1e4).round(4)
    panel["substantive_per_10k_words"] = (
        (panel["n_substantive"] / words * 1e4).where(panel["classification_complete"]).round(4)
    )
    denom = panel["n_keyword_sentences"].replace(0, np.nan)
    panel["substantive_share"] = (panel["n_substantive"] / denom).where(panel["classification_complete"]).round(4)

    _add_breakdowns(panel, sent)
    _add_dynamics(panel, sent, repeat_threshold)
    panel["evidence"] = panel["filing_id"].map(lambda fid: ";".join(evidence.get(fid, [])))

    if firm_attributes is not None and len(firm_attributes):
        attrs = firm_attributes.copy()
        if "firm_id" not in attrs.columns:
            raise ValueError("firm_attributes needs a 'firm_id' column")
        attrs["firm_id"] = attrs["firm_id"].astype(str)
        keys = ["firm_id"]
        if "fiscal_year" in attrs.columns:
            attrs["fiscal_year"] = pd.to_numeric(attrs["fiscal_year"], errors="coerce").astype("Int64")
            keys.append("fiscal_year")
        clash = [c for c in attrs.columns if c in panel.columns and c not in keys]
        if clash:
            attrs = attrs.rename(columns={c: f"attr_{c}" for c in clash})
        if attrs.duplicated(keys).any():
            raise ValueError("firm_attributes must have unique merge keys")
        panel = panel.merge(attrs, on=keys, how="left", validate="many_to_one")
    return panel


def _add_breakdowns(panel: pd.DataFrame, sent: pd.DataFrame) -> None:
    fids = panel["filing_id"]
    zero = pd.Series(0, index=panel.index)
    classified = sent
    if not sent.empty and "classification_status" in sent:
        classified = sent[sent["classification_status"] == "ok"]
    subs = classified[classified["state"] == "SUBSTANTIVE"] if not sent.empty else sent
    weak = classified[classified["state"] == "WEAK"] if not sent.empty else sent
    for cat in CATEGORIES:
        col = f"n_substantive_{cat}"
        if subs.empty:
            panel[col] = zero
        else:
            has = subs["categories"].astype(str).str.split(";").map(lambda cs, cat=cat: cat in cs)
            counts = subs[has].groupby("filing_id").size()
            panel[col] = fids.map(counts).fillna(0).astype(int)
        panel[f"substantive_{cat}_any"] = pd.array(
            [
                1 if count > 0 else (0 if complete else pd.NA)
                for count, complete in zip(panel[col], panel["classification_complete"])
            ],
            dtype="Int64",
        )
    for state_df, prefix, reasons in ((subs, "n_substantive_", SUBSTANTIVE_REASONS), (weak, "n_weak_", WEAK_REASONS)):
        for reason in reasons:
            col = prefix + reason.lower()
            if state_df.empty:
                panel[col] = zero
            else:
                counts = state_df[state_df["reason"] == reason].groupby("filing_id").size()
                panel[col] = fids.map(counts).fillna(0).astype(int)


def _add_dynamics(panel: pd.DataFrame, sent: pd.DataFrame, threshold: float) -> None:
    tokens_by_filing: dict[str, list[frozenset[str]]] = {}
    subs_by_filing: dict[str, list[frozenset[str]]] = {}
    if not sent.empty:
        hit = sent[sent["raw_terms"].astype(str) != ""]
        for fid, grp in hit.groupby("filing_id"):
            toks = [_tokens(t) for t in grp["text"]]
            tokens_by_filing[fid] = toks
            valid = grp.get("classification_status", pd.Series("ok", index=grp.index)) == "ok"
            subs_by_filing[fid] = [t for t, st, ok in zip(toks, grp["state"], valid) if ok and st == "SUBSTANTIVE"]
    filing_of = {
        (f, int(y)): fid for f, y, fid in zip(panel["firm_id"], panel["fiscal_year"], panel["filing_id"]) if pd.notna(y)
    }
    sub_of = {
        (f, int(y)): s
        for f, y, s in zip(panel["firm_id"], panel["fiscal_year"], panel["substantive_any"])
        if pd.notna(y) and pd.notna(s)
    }
    first_sub_of: dict[str, int] = {}
    for (f, y), is_sub in sub_of.items():
        if is_sub == 1:
            first_sub_of[f] = min(first_sub_of.get(f, y), y)
    new_counts, novelty, onset = [], [], []
    for firm, year, fid, n_sub in zip(
        panel["firm_id"], panel["fiscal_year"], panel["filing_id"], panel["n_substantive"]
    ):
        prev_fid = filing_of.get((firm, int(year) - 1)) if pd.notna(year) else None
        mine = subs_by_filing.get(fid, [])
        current_complete = bool(panel.loc[panel["filing_id"] == fid, "classification_complete"].iloc[0])
        prev_complete = (
            bool(panel.loc[panel["filing_id"] == prev_fid, "classification_complete"].iloc[0])
            if prev_fid is not None
            else False
        )
        if prev_fid is None or not current_complete or not prev_complete:
            new_counts.append(pd.NA)
            novelty.append(np.nan)
        else:
            prev = tokens_by_filing.get(prev_fid, [])
            n_new = sum(1 for t in mine if not any(_jaccard(t, p) >= threshold for p in prev))
            new_counts.append(n_new)
            novelty.append(round(n_new / n_sub, 4) if n_sub else np.nan)
        if pd.isna(year) or not current_complete:
            onset.append(pd.NA)
        elif first_sub_of.get(firm) != int(year):
            onset.append(0)  # not the firm's first SUBSTANTIVE year
        elif (firm, int(year) - 1) in sub_of and pd.notna(sub_of[(firm, int(year) - 1)]):
            onset.append(1)  # previous year observed, necessarily without a SUBSTANTIVE claim
        else:
            onset.append(pd.NA)  # censored: the year before the first SUBSTANTIVE year is not observed
    panel["n_substantive_new"] = pd.array(new_counts, dtype="Int64")
    panel["substantive_novelty"] = novelty
    panel["substantive_onset"] = pd.array(onset, dtype="Int64")
    years = pd.to_numeric(panel["fiscal_year"], errors="coerce")
    panel["first_observed_year"] = years.groupby(panel["firm_id"]).transform("min").astype("Int64")
    first_sub = years.where(panel["substantive_any"] == 1).groupby(panel["firm_id"]).transform("min")
    panel["first_substantive_year"] = first_sub.astype("Int64")


def group_panel(
    firm_panel: pd.DataFrame,
    by: Iterable[str] = ("country", "fiscal_year"),
    weight: str | None = None,
    min_firms: int = 1,
) -> pd.DataFrame:
    """Aggregate a firm-year panel to groups such as country-year or industry-year.

    ``weight`` names a numeric column (e.g. employees merged through
    ``firm_attributes``); firms with a missing or non-positive weight are left
    out of weighted means, and ``n_firms_weighted`` reports how many remain.
    """
    by = list(by)
    mean_fields = ("keyword_per_10k_words", "substantive_per_10k_words", "substantive_share")

    def empty_panel():
        cols = by + [
            "n_firms",
            "n_firms_complete",
            "n_firms_incomplete",
            "n_firms_claim_observed",
            "n_firms_substantive_observed",
            "share_keyword",
            "share_keyword_substantive_observed",
            "share_claim",
            "share_substantive",
            "share_claim_observed",
            "share_substantive_observed",
        ]
        for field in mean_fields:
            cols += ["mean_" + field, "mean_" + field + "_observed", "n_firms_" + field + "_observed"]
        cols += [
            "share_claim_lower_bound",
            "share_claim_upper_bound",
            "share_substantive_lower_bound",
            "share_substantive_upper_bound",
        ]
        if weight is not None:
            cols.append("n_firms_weighted")
        cols += ["share_substantive_" + c for c in CATEGORIES if f"substantive_{c}_any" in firm_panel.columns]
        if "substantive_novelty" in firm_panel.columns:
            cols += ["mean_substantive_novelty", "share_onset"]
        cols += ["gap_keyword_substantive_pp"]
        return pd.DataFrame(columns=list(dict.fromkeys(cols)))

    if firm_panel.empty:
        return empty_panel()
    df = firm_panel.copy()
    if weight is not None and weight not in df.columns:
        raise ValueError(f"weight column {weight!r} is not in the panel")
    df["_w"] = 1.0 if weight is None else pd.to_numeric(df[weight], errors="coerce").fillna(0.0).clip(lower=0.0)
    if not np.isfinite(df["_w"]).all():
        raise ValueError("group weights must be finite")
    cat_cols = [f"substantive_{c}_any" for c in CATEGORIES if f"substantive_{c}_any" in df.columns]
    rows = []
    for key, g in df.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        w = g["_w"].to_numpy(dtype=float)
        wsum = w.sum()
        if len(g) < min_firms or wsum <= 0:
            continue

        def wmean(col: str, g=g, w=w) -> float:
            if col not in g.columns:
                return float("nan")
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            mask = np.isfinite(vals)
            if not mask.any() or w[mask].sum() == 0:
                return float("nan")
            return float(np.average(vals[mask], weights=w[mask]))

        def complete_share(col: str, g=g, w=w, wmean=wmean) -> float:
            if col not in g.columns:
                return float("nan")
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            return float("nan") if np.any(~np.isfinite(vals) & (w > 0)) else wmean(col)

        def bounds(col: str, g=g, w=w, wsum=wsum) -> tuple[float, float]:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if np.any(~np.isnan(vals) & ~np.isin(vals, [0, 1])):
                raise ValueError(f"{col} must contain binary values or missing")
            low = np.where(np.isnan(vals), 0.0, vals)
            high = np.where(np.isnan(vals), 1.0, vals)
            return float(np.dot(w, low) / wsum), float(np.dot(w, high) / wsum)

        row = dict(zip(by, key))
        observed_substantive = g[g["substantive_any"].notna()]
        observed_weights = observed_substantive["_w"].to_numpy(dtype=float)
        row.update(
            n_firms=int(len(g)),
            n_firms_complete=int(g["classification_complete"].sum()),
            n_firms_incomplete=int((~g["classification_complete"]).sum()),
            n_firms_claim_observed=int(g["claim_any"].notna().sum()),
            n_firms_substantive_observed=int(g["substantive_any"].notna().sum()),
            share_keyword=wmean("keyword_any"),
            share_keyword_substantive_observed=wmean("keyword_any", g=observed_substantive, w=observed_weights),
            share_claim=complete_share("claim_any"),
            share_substantive=complete_share("substantive_any"),
            share_claim_observed=wmean("claim_any"),
            share_substantive_observed=wmean("substantive_any"),
        )
        for col in mean_fields:
            row["mean_" + col] = complete_share(col)
            row["mean_" + col + "_observed"] = wmean(col)
            vals = (
                pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                if col in g
                else np.full(len(g), np.nan)
            )
            row["n_firms_" + col + "_observed"] = int((np.isfinite(vals) & (w > 0)).sum())
        row["share_claim_lower_bound"], row["share_claim_upper_bound"] = bounds("claim_any")
        row["share_substantive_lower_bound"], row["share_substantive_upper_bound"] = bounds("substantive_any")
        if weight is not None:
            row["n_firms_weighted"] = int((w > 0).sum())
        for col in cat_cols:
            row["share_" + col.removesuffix("_any")] = complete_share(col)
        if "substantive_novelty" in g.columns:
            row["mean_substantive_novelty"] = wmean("substantive_novelty")
            row["share_onset"] = wmean("substantive_onset")
        row["gap_keyword_substantive_pp"] = 100.0 * (
            row["share_keyword_substantive_observed"] - row["share_substantive"]
        )
        rows.append(row)
    if not rows:
        return empty_panel()
    out = pd.DataFrame(rows)
    num = [c for c in out.columns if c.startswith(("share_", "mean_", "gap_"))]
    out[num] = out[num].round(4)
    return out


def state_mix(results, by: Iterable[str] = ("fiscal_year",)) -> pd.DataFrame:
    """Share of keyword sentences by state and reason (what keyword counts are made of)."""
    s = results.sentences
    if s.empty:
        return pd.DataFrame()
    s = s[s["raw_terms"].astype(str) != ""]
    if "classification_status" in s:
        s = s[s["classification_status"] == "ok"]
    by = list(by)
    counts = s.groupby(by + ["state", "reason"], dropna=False).size().rename("n").reset_index()
    totals = counts.groupby(by, dropna=False)["n"].transform("sum")
    counts["share"] = (counts["n"] / totals).round(4)
    return counts


def read_industry_map(path) -> dict[str, str]:
    """Read a CSV with columns ``industry_code`` and ``group`` (or ``nace_section``)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    target = "group" if "group" in df.columns else "nace_section"
    if "industry_code" not in df.columns or target not in df.columns:
        raise ValueError("industry map CSV needs columns 'industry_code' and 'group' (or 'nace_section')")
    return dict(zip(df["industry_code"].str.strip(), df[target].str.strip()))
