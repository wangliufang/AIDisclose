"""Quality checks for a results directory (``aidisclose qa``).

The checks look for the usual ways a large run goes wrong without failing:
documents that were downloaded but yielded no text, filings whose Items could
not be found, sentences that are too long (a segmentation problem), unusually
frequent identical sentences (boilerplate or a repeated running header),
classifier errors, firm-years with several reports, and gaps in firm
coverage. Each finding has a severity (``warning`` or ``info``) and the rows
it concerns, so that problems can be inspected before indicators are used.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd


def _finding(check: str, severity: str, message: str, n: int, examples: list | None = None) -> dict[str, Any]:
    return {"check": check, "severity": severity, "message": message, "n": int(n), "examples": examples or []}


def quality_report(
    results,
    min_words: int = 2000,
    long_sentence_chars: int = 1000,
    duplicate_min_filings: int = 5,
) -> dict[str, Any]:
    """Run the checks and return ``{"summary": ..., "findings": [...]}``."""
    docs = results.documents.copy()
    sent = results.sentences.copy()
    findings: list[dict[str, Any]] = []
    status = Counter(docs["status"].astype(str))
    ok = docs[docs["status"] == "ok"]

    err = docs[docs["status"] == "error"]
    if len(err):
        msgs = Counter(str(e).split(":")[0] for e in err["error"])
        findings.append(
            _finding(
                "failed_documents",
                "warning",
                f"{len(err)} documents failed ({dict(msgs.most_common(3))}); rerun with --resume to retry them",
                len(err),
                err["filing_id"].head(10).tolist(),
            )
        )
    skipped = docs[docs["status"].astype(str).str.startswith("skipped")]
    if len(skipped):
        langs = Counter(skipped["language"].astype(str).replace("", "unknown"))
        findings.append(
            _finding(
                "skipped_language",
                "info",
                f"{len(skipped)} documents skipped for language {dict(langs.most_common(5))}; check that the "
                "skipped share does not differ across the groups you compare",
                len(skipped),
                skipped["filing_id"].head(10).tolist(),
            )
        )
    n_words = pd.to_numeric(ok["n_words"], errors="coerce").fillna(0)
    empty = ok[n_words == 0]
    if len(empty):
        findings.append(
            _finding(
                "empty_text",
                "warning",
                f"{len(empty)} documents yielded no text (scanned PDF or wrong file?)",
                len(empty),
                empty["filing_id"].head(10).tolist(),
            )
        )
    short = ok[(n_words > 0) & (n_words < min_words)]
    if len(short):
        findings.append(
            _finding(
                "short_documents",
                "info",
                f"{len(short)} documents have fewer than {min_words} words (cover documents, wrappers, extracts?)",
                len(short),
                short["filing_id"].head(10).tolist(),
            )
        )
    if not sent.empty:
        sec = sent.groupby("filing_id")["section"].agg(lambda s: set(s))
        forms = ok.set_index("filing_id")["form"].astype(str) if "form" in ok else pd.Series(dtype=str)
        item_forms = forms[forms.str.upper().str.startswith(("10-K", "20-F"))]
        unsplit = [
            fid
            for fid in item_forms.index
            if fid in sec.index and all(s in ("full",) or s.endswith(":main") for s in sec[fid])
        ]
        if unsplit:
            findings.append(
                _finding(
                    "items_not_found",
                    "warning",
                    f"{len(unsplit)} 10-K/20-F documents were not split into Items; section filters do not "
                    "apply to them",
                    len(unsplit),
                    unsplit[:10],
                )
            )
        lengths = sent["text"].astype(str).str.len()
        long = sent[lengths >= long_sentence_chars]
        if len(long):
            findings.append(
                _finding(
                    "long_sentences",
                    "warning" if len(long) / max(len(sent), 1) > 0.02 else "info",
                    f"{len(long)} sentences have {long_sentence_chars}+ characters (tables or lists read as one "
                    "sentence?)",
                    len(long),
                    long["sentence_id"].head(10).tolist(),
                )
            )
        norm = sent["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        per_text = sent.assign(_t=norm).groupby("_t")["filing_id"].nunique()
        common = per_text[per_text >= duplicate_min_filings].sort_values(ascending=False)
        if len(common):
            findings.append(
                _finding(
                    "repeated_sentences",
                    "info",
                    f"{len(common)} distinct sentences occur in {duplicate_min_filings}+ filings (boilerplate or "
                    "running headers); the most frequent is shown first",
                    len(common),
                    [f"{n} filings: {t[:120]}" for t, n in common.head(5).items()],
                )
            )
        fp = sent[(sent["raw_terms"].astype(str) != "") & (sent["reason"] == "FALSE_POSITIVE")]
        if len(fp):
            terms = Counter(t for ts in fp["raw_terms"].astype(str) for t in ts.split(";") if t)
            findings.append(
                _finding(
                    "false_positives",
                    "info",
                    f"{len(fp)} keyword sentences were rejected as false positives; most frequent terms "
                    f"{dict(terms.most_common(5))}",
                    len(fp),
                    fp["sentence_id"].head(5).tolist(),
                )
            )
    n_err = int(pd.to_numeric(docs.get("n_classifier_errors", 0), errors="coerce").fillna(0).sum())
    if n_err:
        findings.append(
            _finding(
                "classifier_errors",
                "warning",
                f"{n_err} sentences have missing classifier labels and are excluded from adoption indicators",
                n_err,
            )
        )
    if len(ok):
        fy = pd.to_numeric(ok["fiscal_year"], errors="coerce")
        no_year = ok[fy.isna()]
        if len(no_year):
            findings.append(
                _finding(
                    "missing_fiscal_year",
                    "warning",
                    f"{len(no_year)} documents have no fiscal year and are dropped from panels",
                    len(no_year),
                    no_year["filing_id"].head(10).tolist(),
                )
            )
        dup = ok.assign(_y=fy).dropna(subset=["_y"]).groupby(["firm_id", "_y"]).size()
        dup = dup[dup > 1]
        if len(dup):
            findings.append(
                _finding(
                    "several_reports_per_firm_year",
                    "info",
                    f"{len(dup)} firm-years have several reports; panels keep the longest",
                    len(dup),
                    [f"{f} {int(y)}" for f, y in dup.index[:10]],
                )
            )
        gaps = []
        for firm, years in ok.assign(_y=fy).dropna(subset=["_y"]).groupby("firm_id")["_y"]:
            ys = sorted(set(int(y) for y in years))
            missing = sorted(set(range(ys[0], ys[-1] + 1)) - set(ys))
            if missing:
                gaps.append(f"{firm}: {missing}")
        if gaps:
            findings.append(
                _finding(
                    "coverage_gaps",
                    "info",
                    f"{len(gaps)} firms have missing years inside their range "
                    "(novelty and onset are not computed across a gap)",
                    len(gaps),
                    gaps[:10],
                )
            )
    if getattr(results, "recall_pool", None) is None or len(results.recall_pool) == 0:
        findings.append(
            _finding(
                "no_recall_pool",
                "info",
                "no recall pool: lexicon recall cannot be audited for this run (use --recall-random and "
                "--recall-probe-cap)",
                0,
            )
        )
    summary = {
        "n_documents": int(len(docs)),
        "status": dict(status),
        "n_sentences": int(len(sent)),
        "n_warnings": sum(1 for f in findings if f["severity"] == "warning"),
        "n_info": sum(1 for f in findings if f["severity"] == "info"),
        "provenance_version": (results.provenance or {}).get("aidisclose_version"),
    }
    return {"summary": summary, "findings": findings}


def quality_to_markdown(rep: dict[str, Any]) -> str:
    s = rep["summary"]
    lines = [
        "# Quality report",
        "",
        f"{s['n_documents']} documents ({json.dumps(s['status'])}), {s['n_sentences']} keyword sentences; "
        f"{s['n_warnings']} warnings, {s['n_info']} notes.",
        "",
    ]
    if not rep["findings"]:
        lines.append("No findings.")
    for f in sorted(rep["findings"], key=lambda f: f["severity"] != "warning"):
        lines.append(f"- **{f['severity'].upper()}** `{f['check']}`: {f['message']}")
        for ex in f["examples"][:5]:
            lines.append(f"    - {ex}")
    return "\n".join(lines) + "\n"
