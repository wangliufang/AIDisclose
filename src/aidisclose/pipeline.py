"""End-to-end measurement pipeline: documents -> candidate sentences -> labels.

Pipeline features
-----------------
* **Checkpoints and resume.** ``Pipeline.run(..., checkpoint_dir=d)`` writes one
  file per processed filing; ``resume=True`` skips filings already on disk, so
  an interrupted run over thousands of reports continues where it stopped.
  The run configuration is fingerprinted and a resume with a different
  configuration is refused.
* **Parallel workers.** ``workers=N`` processes filings concurrently in
  threads (``executor="thread"``, best for downloads and LLM calls) or in
  processes (``executor="process"``, best for CPU-bound extraction of local
  files). In process mode source HTTP clients share a request clock across
  workers within the run; other runs and applications are not coordinated.
* **Recall pool.** With ``recall_random > 0`` or ``recall_probe_cap > 0`` the
  pipeline keeps a reproducible random sample of sentences that the lexicons
  did *not* retrieve, stratified by a probe lexicon of AI-adjacent terms, with
  their inclusion probabilities. Annotating a sample of this pool estimates
  how many AI claims the lexicon misses (``aidisclose sample-recall`` and
  ``evaluate-recall``).
* **Language routing.** ``languages`` and ``lexicons`` route documents to a
  lexicon per language. AIDisclose ships English lexicons only; the routing is
  infrastructure for user-supplied, validated lexicons in other languages
  (with the ``supervised`` or ``llm`` backend, since the rule backend's cue
  words are English).
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import hashlib
import hmac
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import platform
import random
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from ._version import __version__
from .classify import get_classifier
from .classify.base import Classifier
from .lexicon import Lexicon, conflicting_acronyms
from .models import AdoptionState, Candidate, Document, Filing, Reason
from .text.language import detect_language
from .text.sections import SectionFilter
from .text.segment import split_sentences

log = logging.getLogger(__name__)

SENTENCE_COLUMNS = [
    "sentence_id",
    "filing_id",
    "firm_id",
    "firm_name",
    "fiscal_year",
    "country",
    "source",
    "section",
    "text",
    "prev_text",
    "next_text",
    "terms",
    "term_spans",
    "categories",
    "raw_terms",
    "filtered_terms",
    "baseline_terms",
    "state",
    "reason",
    "classification_status",
    "confidence",
    "backend",
    "details",
]
DOCUMENT_COLUMNS = [
    "filing_id",
    "firm_id",
    "firm_name",
    "fiscal_year",
    "period_end",
    "country",
    "industry_code",
    "industry_scheme",
    "source",
    "form",
    "url",
    "language",
    "status",
    "error",
    "n_words",
    "n_sentences",
    "n_keyword_sentences",
    "n_baseline_sentences",
    "n_candidates",
    "n_substantive",
    "n_weak",
    "n_no_claim",
    "n_false_positive",
    "n_classifier_errors",
    "n_probe_sentences",
    "n_other_sentences",
    "extra",
]
RECALL_COLUMNS = [
    "sentence_id",
    "filing_id",
    "firm_id",
    "firm_name",
    "fiscal_year",
    "country",
    "section",
    "text",
    "prev_text",
    "next_text",
    "pool",
    "probe_terms",
    "stratum_size",
    "n_drawn",
    "inclusion_prob",
]
_VOLATILE_PROVENANCE = {"created_utc", "python", "platform"}


@dataclass
class Results:
    """Sentence- and document-level output of a pipeline run.

    ``recall_pool`` holds the sentences kept for the recall audit (empty unless
    the pipeline was configured to keep them).
    """

    sentences: pd.DataFrame
    documents: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    recall_pool: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RECALL_COLUMNS))

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        # Marker makes interrupted replacement detectable even for older result folders.
        marker = out / ".save_in_progress"
        marker.write_text("incomplete result save", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix=".results-", dir=out) as tmp:
            stage = Path(tmp)
            self.sentences.to_csv(stage / "sentences.csv", index=False)
            self.documents.to_csv(stage / "documents.csv", index=False)
            self.recall_pool.reindex(columns=RECALL_COLUMNS).to_csv(stage / "recall_pool.csv", index=False)
            names = ("sentences.csv", "documents.csv", "recall_pool.csv")
            prov = dict(self.provenance)
            prov["result_file_sha256"] = {
                name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in names
            }
            (stage / "provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
            for name in (*names, "provenance.json"):
                os.replace(stage / name, out / name)
        marker.unlink()
        return out

    @classmethod
    def load(cls, directory: str | Path, *, allow_unverified: bool = False) -> Results:
        d = Path(directory)
        if (d / ".save_in_progress").exists():
            raise ValueError("result save was interrupted; rerun or resave before loading")
        prov_path = d / "provenance.json"
        if not prov_path.is_file() and not allow_unverified:
            raise ValueError("result provenance is missing; integrity cannot be verified")
        provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
        checksums = provenance.pop("result_file_sha256", None)
        if checksums is None and not allow_unverified:
            raise ValueError(
                "result has no integrity manifest; import legacy data explicitly with allow_unverified=True"
            )
        if checksums is not None:
            if set(checksums) != {"sentences.csv", "documents.csv", "recall_pool.csv"}:
                raise ValueError("invalid result checksum manifest")
            for name, expected in checksums.items():
                path = d / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"result file missing or modified: {name}")
        # Identifiers, codes and text must never be inferred as numbers: e.g.
        # a firm ID of "000123" is not interchangeable with integer 123.
        # Numeric columns are converted explicitly below after reading.
        sentences = pd.read_csv(d / "sentences.csv", keep_default_na=False, dtype=str)
        documents = pd.read_csv(d / "documents.csv", keep_default_na=False, dtype=str)
        for frame in (sentences, documents):
            if "fiscal_year" in frame:
                frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="raise").astype("Int64")
        for col in ("confidence",):
            if col in sentences:
                sentences[col] = pd.to_numeric(sentences[col], errors="coerce")
        for col in ("n_substantive", "n_weak", "n_no_claim", "n_false_positive", "n_classifier_errors"):
            if col not in documents:
                documents[col] = 0
        for col in [c for c in DOCUMENT_COLUMNS if c.startswith("n_")]:
            if col in documents:
                documents[col] = pd.to_numeric(documents[col], errors="coerce").fillna(0).astype(int)
            else:  # results written by an earlier version
                documents[col] = 0
        if "classification_status" not in sentences:
            # Read legacy result files without treating failed LLM replies as
            # genuine NO_CLAIM labels.
            statuses = []
            for details in sentences.get("details", pd.Series("", index=sentences.index)).astype(str):
                try:
                    is_error = "error" in json.loads(details or "{}")
                except (json.JSONDecodeError, TypeError):
                    is_error = False
                statuses.append("error" if is_error else "ok")
            sentences["classification_status"] = statuses
            failed = sentences["classification_status"] == "error"
            sentences.loc[failed, ["state", "reason"]] = ""
        # Recount from sentence rows so legacy files no longer include failed
        # predictions in n_no_claim (or any other state total).
        if not sentences.empty and "filing_id" in sentences:
            valid = sentences[(sentences["classification_status"] == "ok") & (sentences["raw_terms"].astype(str) != "")]
            for filing_id, doc_ix in documents.groupby("filing_id", sort=False).groups.items():
                rows = sentences[sentences["filing_id"] == filing_id]
                states = valid.loc[valid["filing_id"] == filing_id, "state"].astype(str)
                counts = {
                    "n_substantive": int((states == "SUBSTANTIVE").sum()),
                    "n_weak": int((states == "WEAK").sum()),
                    "n_no_claim": int((states == "NO_CLAIM").sum()),
                    "n_false_positive": int(
                        ((rows["reason"] == "FALSE_POSITIVE") & (rows["raw_terms"].astype(str) != "")).sum()
                    ),
                    "n_classifier_errors": int((rows["classification_status"] == "error").sum()),
                }
                for col, count in counts.items():
                    documents.loc[doc_ix, col] = count
        pool_path = d / "recall_pool.csv"
        if pool_path.exists():
            pool = pd.read_csv(pool_path, keep_default_na=False, dtype=str)
            if "fiscal_year" in pool:
                pool["fiscal_year"] = pd.to_numeric(pool["fiscal_year"], errors="raise").astype("Int64")
            for col in ("stratum_size", "n_drawn"):
                pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0).astype(int)
            pool["inclusion_prob"] = pd.to_numeric(pool["inclusion_prob"], errors="coerce")
        else:
            pool = pd.DataFrame(columns=RECALL_COLUMNS)
        return cls(sentences=sentences, documents=documents, provenance=provenance, recall_pool=pool)

    def dataset_digest(self) -> str:
        """Bind samples to the exact loaded sentence, document and recall tables."""
        h = hashlib.sha256()
        for name, frame in (
            ("sentences", self.sentences),
            ("documents", self.documents),
            ("recall_pool", self.recall_pool),
        ):
            payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            h.update(name.encode("utf-8"))
            h.update(len(payload).to_bytes(8, "big"))
            h.update(payload)
        return h.hexdigest()

    @classmethod
    def concat(cls, parts: Sequence[Results]) -> Results:
        if not parts:
            raise ValueError("nothing to concatenate")

        def signature(prov):
            return {
                k: v
                for k, v in prov.items()
                if k
                not in {
                    "created_utc",
                    "input_manifest_sha256",
                    "n_input_filings",
                    "source",
                    "combined_runs",
                    "relabelled_utc",
                }
            }

        signatures = [json.dumps(signature(p.provenance), sort_keys=True, default=str) for p in parts]
        if len(set(signatures)) > 1:
            raise ValueError("results use different measurement configurations; relabel or rerun before concatenating")
        all_filings = pd.concat([p.documents["filing_id"] for p in parts], ignore_index=True)
        if all_filings.duplicated().any():
            raise ValueError("results contain duplicate filing identifiers; choose a non-overlapping input set")
        pools = [p.recall_pool for p in parts if len(p.recall_pool)]
        provenance = dict(parts[0].provenance)
        provenance["combined_runs"] = []
        for part in parts:
            previous = part.provenance.get("combined_runs")
            if previous is not None:
                if not isinstance(previous, list):
                    raise ValueError("invalid combined run provenance")
                provenance["combined_runs"].extend(previous)
            else:
                provenance["combined_runs"].append(
                    {
                        "source": part.provenance.get("source"),
                        "input_manifest_sha256": part.provenance.get("input_manifest_sha256"),
                        "n_input_filings": part.provenance.get("n_input_filings"),
                    }
                )
        # A constituent digest does not identify the combined input.
        provenance.pop("input_manifest_sha256", None)
        provenance["n_input_filings"] = len(all_filings)
        provenance["source"] = {"name": "combined-runs"}
        return cls(
            sentences=pd.concat([p.sentences for p in parts], ignore_index=True),
            documents=pd.concat([p.documents for p in parts], ignore_index=True),
            provenance=provenance,
            recall_pool=pd.concat(pools, ignore_index=True) if pools else pd.DataFrame(columns=RECALL_COLUMNS),
        )

    def evidence(
        self, firm_id: str | None = None, fiscal_year: int | None = None, state: str = "SUBSTANTIVE"
    ) -> pd.DataFrame:
        """Return the sentences behind an indicator value (the evidence trail)."""
        df = self.sentences
        mask = df["state"] == state
        if firm_id is not None:
            mask &= df["firm_id"] == firm_id
        if fiscal_year is not None:
            mask &= pd.to_numeric(df["fiscal_year"], errors="coerce") == fiscal_year
        cols = ["sentence_id", "firm_name", "fiscal_year", "section", "text", "state", "reason", "confidence"]
        return df.loc[mask, cols].reset_index(drop=True)


@dataclass
class DocumentOutcome:
    """Everything the pipeline produces for one filing."""

    rows: list[dict[str, Any]]
    doc_row: dict[str, Any]
    recall_rows: list[dict[str, Any]] = field(default_factory=list)


class Pipeline:
    """Measure AI adoption claims in disclosure documents.

    Parameters
    ----------
    backend:
        ``"rules"`` (default), ``"supervised"``, ``"llm"`` or a
        :class:`~aidisclose.classify.base.Classifier` instance.
    lexicon:
        Built-in lexicon name, path to a YAML lexicon or a :class:`Lexicon`
        (used for English, and for every language when ``languages`` is
        ``None`` and ``english_only`` is ``False``).
    tiers:
        Lexicon tiers to use (``("core",)`` by default, ``None`` for all).
    baseline_lexicon:
        Lexicon used to compute the plain keyword baseline (English).
    sections:
        Section labels to process (e.g. ``["10-K:1", "10-K:1A", "10-K:7"]``)
        or a preset name (``"core"``, ``"narrative"``). A filter restricts
        only the document types it names; documents without sections are
        always processed in full (see :class:`~aidisclose.text.sections.SectionFilter`).
    english_only:
        Skip documents whose detected language is not English (ignored when
        ``languages`` is given).
    languages:
        Languages to process, e.g. ``("en", "de")``. Every language other than
        English needs an entry in ``lexicons``.
    lexicons:
        ``{language: lexicon}`` for languages other than English.
    baseline_lexicons:
        ``{language: lexicon}`` keyword baselines for other languages (default:
        the language's measurement lexicon).
    recall_random, recall_probe_cap, recall_seed, probe_lexicon:
        Recall pool: per document, keep up to ``recall_probe_cap`` sentences
        that match the probe lexicon but not the measurement lexicon, and
        ``recall_random`` of the remaining non-retrieved sentences, drawn with
        a seed derived from ``recall_seed`` and the filing id.
    backend_options:
        Keyword arguments passed to the backend constructor.
    """

    def __init__(
        self,
        backend: str | Classifier = "rules",
        lexicon: str | Path | Lexicon = "default",
        tiers: Iterable[str] | None = ("core",),
        baseline_lexicon: str | Path | Lexicon = "keyword-baseline-4",
        sections: Iterable[str] | str | None = None,
        english_only: bool = True,
        backend_options: dict[str, Any] | None = None,
        max_sentence_chars: int = 1200,
        languages: Iterable[str] | None = None,
        lexicons: Mapping[str, str | Path | Lexicon] | None = None,
        baseline_lexicons: Mapping[str, str | Path | Lexicon] | None = None,
        recall_random: int = 0,
        recall_probe_cap: int = 0,
        recall_seed: int = 20240601,
        probe_lexicon: str | Path | Lexicon = "recall-probe",
    ) -> None:
        def load(x, t):
            return x if isinstance(x, Lexicon) else Lexicon.load(x, tiers=t)

        self.lexicon = load(lexicon, tiers)
        self.baseline = load(baseline_lexicon, None)
        self.classifier = (
            backend if isinstance(backend, Classifier) else get_classifier(backend, **(backend_options or {}))
        )
        self.section_filter = SectionFilter(sections)
        self.sections = self.section_filter.labels or None
        self.max_sentence_chars = max_sentence_chars
        if languages is not None:
            self.languages: tuple[str, ...] | None = tuple(dict.fromkeys(lang.lower() for lang in languages))
        else:
            self.languages = ("en",) if english_only else None
        self.english_only = self.languages == ("en",)
        self.lexicons: dict[str, Lexicon] = {"en": self.lexicon}
        for lang, lex in (lexicons or {}).items():
            self.lexicons[lang.lower()] = load(lex, tiers)
        self.baselines: dict[str, Lexicon] = {"en": self.baseline}
        for lang, lex in (baseline_lexicons or {}).items():
            self.baselines[lang.lower()] = load(lex, None)
        if self.languages is not None:
            missing = [lang for lang in self.languages if lang not in self.lexicons]
            if missing:
                raise ValueError(
                    f"no lexicon for language(s) {missing}; AIDisclose ships English lexicons only, pass "
                    "lexicons={'<language>': '<path to a validated YAML lexicon>'}"
                )
        if self.classifier.name == "rules" and any(lang != "en" for lang in self.lexicons):
            raise ValueError(
                "the rules backend uses English cue words; use the supervised or llm backend for other languages"
            )
        self.recall_random = int(recall_random)
        self.recall_probe_cap = int(recall_probe_cap)
        self.recall_seed = int(recall_seed)
        self.probe = load(probe_lexicon, None) if (self.recall_random or self.recall_probe_cap) else None

    # ------------------------------------------------------------- provenance
    def provenance(self) -> dict[str, Any]:
        prov = {
            "aidisclose_version": __version__,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lexicon": self.lexicon.describe(),
            "baseline_lexicon": self.baseline.describe(),
            "classifier": self.classifier.config(),
            "sections": self.sections,
            "english_only": self.english_only,
            "languages": list(self.languages) if self.languages else None,
            "max_sentence_chars": self.max_sentence_chars,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "requests", "lxml", "beautifulsoup4", "PyYAML")
                if _distribution_exists(name)
            },
        }
        other = {k: v.describe() for k, v in self.lexicons.items() if k != "en"}
        if other:
            prov["lexicons"] = other
            prov["baseline_lexicons"] = {k: v.describe() for k, v in self.baselines.items() if k != "en"}
        if self.probe is not None:
            prov["recall_pool"] = {
                "random_per_document": self.recall_random,
                "probe_cap_per_document": self.recall_probe_cap,
                "seed": self.recall_seed,
                "probe_lexicon": self.probe.describe(),
            }
        return prov

    def fingerprint(self, source_config: Mapping[str, Any] | None = None) -> str:
        """Digest of the configuration that determines the output (for checkpoints)."""
        stable = {k: v for k, v in self.provenance().items() if k not in _VOLATILE_PROVENANCE}
        if source_config is not None:
            stable["source"] = dict(source_config)
        return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------- processing
    def _lexicons_for(self, lang: str | None) -> tuple[Lexicon, Lexicon]:
        lang = (lang or "en").lower()
        lex = self.lexicons.get(lang)
        if lex is None:
            raise KeyError(f"no lexicon configured for detected language {lang!r}")
        base = self.baselines.get(lang) or (self.baseline if lang == "en" else lex)
        return lex, base

    def extract_candidates(self, doc: Document) -> tuple[list[Candidate], dict[str, int]]:
        """Split a document into sentences and keep those with lexicon hits."""
        candidates, counts, _ = self._extract(doc)
        return candidates, counts

    def _extract(self, doc: Document):
        lang = (doc.language or "en").lower()
        lexicon, baseline = self._lexicons_for(lang)
        disabled = conflicting_acronyms(doc.text) if lang == "en" else set()
        f = doc.filing
        candidates: list[Candidate] = []
        counts = {"n_sentences": 0, "n_keyword_sentences": 0, "n_baseline_sentences": 0}
        keep_pool = self.probe is not None
        probe_pool: list[tuple[str, str, int, list[str], list]] = []
        other_pool: list[tuple[str, str, int, list[str], list]] = []
        for section, text in doc.iter_sections():
            if doc.sections and not self.section_filter(section):
                continue
            spans = split_sentences(text, max_chars=self.max_sentence_chars)
            counts["n_sentences"] += len(spans)
            for i, sp in enumerate(spans):
                hits = lexicon.find(sp.text, disabled_terms=disabled)
                base = baseline.find(sp.text)
                # Recall is for the main measurement lexicon. A baseline-only
                # hit is not in its retrieved population and must remain eligible.
                if not hits.raw and keep_pool:
                    probe_terms = self.probe.find(sp.text).term_ids("raw") if (lang == "en" and self.probe) else []
                    (probe_pool if probe_terms else other_pool).append((section, sp.text, i, probe_terms, spans))
                if not hits.raw and not base.raw:
                    continue
                counts["n_keyword_sentences"] += int(bool(hits.raw))
                counts["n_baseline_sentences"] += int(bool(base.raw))
                prev_text, next_text = _context(spans, i)
                candidates.append(
                    Candidate(
                        filing_id=f.filing_id,
                        # stable across backends, lexicon tiers and section filters
                        sentence_id=f"{f.filing_id}#{section}#{i:05d}",
                        section=section,
                        text=sp.text,
                        prev_text=prev_text,
                        next_text=next_text,
                        firm_name=f.firm_name,
                        terms=hits.term_ids("valid"),
                        term_spans=[(h.start, h.end) for h in hits.valid],
                        categories=hits.categories(),
                        raw_terms=hits.term_ids("raw"),
                        filtered=hits.term_ids("filtered"),
                        filter_reasons=dict(hits.reasons),
                        baseline_terms=base.term_ids("raw"),
                    )
                )
        recall_rows: list[dict[str, Any]] = []
        if keep_pool:
            counts["n_probe_sentences"] = len(probe_pool)
            counts["n_other_sentences"] = len(other_pool)
            rng = random.Random(f"{self.recall_seed}:{f.filing_id}")
            strata = (("probe", probe_pool, self.recall_probe_cap), ("random", other_pool, self.recall_random))
            for name, pool, k in strata:
                if not pool or k <= 0:
                    continue
                drawn = sorted(rng.sample(range(len(pool)), min(k, len(pool))))
                for j in drawn:
                    section, text, i, probe_terms, spans = pool[j]
                    prev_text, next_text = _context(spans, i)
                    recall_rows.append(
                        {
                            "sentence_id": f"{f.filing_id}#{section}#{i:05d}",
                            "filing_id": f.filing_id,
                            "firm_id": f.firm_id,
                            "firm_name": f.firm_name,
                            "fiscal_year": f.fiscal_year,
                            "country": f.country or "",
                            "section": section,
                            "text": text,
                            "prev_text": prev_text,
                            "next_text": next_text,
                            "pool": name,
                            "probe_terms": ";".join(probe_terms),
                            "stratum_size": len(pool),
                            "n_drawn": len(drawn),
                            "inclusion_prob": len(drawn) / len(pool),
                        }
                    )
        return candidates, counts, recall_rows

    def process_document(self, doc: Document) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Classify one document; returns ``(sentence rows, document row)``."""
        out = self.analyse(doc)
        return out.rows, out.doc_row

    def analyse(self, doc: Document) -> DocumentOutcome:
        """Classify one document, including its recall-pool sample."""
        f = doc.filing
        doc_row = self._doc_row(f)
        lang = doc.language
        if lang is None:
            lang, _ = detect_language(doc.text)
            doc = replace(doc, language=lang)
        doc_row["language"] = lang or ""
        doc_row["n_words"] = doc.n_words
        if self.languages is not None and lang not in self.languages:
            doc_row["status"] = "skipped_non_english" if self.languages == ("en",) else "skipped_language"
            return DocumentOutcome([], doc_row)
        if lang and lang.lower() not in self.lexicons:
            doc_row["status"] = "skipped_no_lexicon"
            doc_row["error"] = f"no lexicon configured for detected language {lang!r}"
            return DocumentOutcome([], doc_row)
        candidates, counts, recall_rows = self._extract(doc)
        doc_row.update(counts)
        # Sentences with only baseline hits (e.g. "ML" without co-occurrence
        # that the main lexicon rejected) are false positives by construction.
        to_classify = [c for c in candidates if c.terms]
        preds = self.classifier.classify(to_classify) if to_classify else []
        if len(preds) != len(to_classify):
            raise RuntimeError(f"classifier returned {len(preds)} predictions for {len(to_classify)} candidates")
        pred_by_id = {c.sentence_id: p for c, p in zip(to_classify, preds)}
        rows = []
        for c in candidates:
            p = pred_by_id.get(c.sentence_id)
            if p is None:
                row_pred = {
                    "state": AdoptionState.NO_CLAIM.value,
                    "reason": Reason.FALSE_POSITIVE.value,
                    "classification_status": "ok",
                    "confidence": 0.8,
                    "backend": "lexicon",
                    "details": json.dumps({"filtered_terms": c.filtered, "filter_reasons": c.filter_reasons}),
                }
            else:
                row_pred = p.as_row()
            rows.append(
                {
                    "sentence_id": c.sentence_id,
                    "filing_id": f.filing_id,
                    "firm_id": f.firm_id,
                    "firm_name": f.firm_name,
                    "fiscal_year": f.fiscal_year,
                    "country": f.country or "",
                    "source": f.source,
                    "section": c.section,
                    "text": c.text,
                    "prev_text": c.prev_text,
                    "next_text": c.next_text,
                    "terms": ";".join(c.terms),
                    "term_spans": json.dumps(c.term_spans, separators=(",", ":")),
                    "categories": ";".join(c.categories),
                    "raw_terms": ";".join(c.raw_terms),
                    "filtered_terms": ";".join(c.filtered),
                    "baseline_terms": ";".join(c.baseline_terms),
                    **row_pred,
                }
            )
        states = [r["state"] for r in rows if r["raw_terms"] and r["state"] in {s.value for s in AdoptionState}]
        doc_row["n_candidates"] = len(to_classify)
        doc_row["n_substantive"] = states.count("SUBSTANTIVE")
        doc_row["n_weak"] = states.count("WEAK")
        doc_row["n_no_claim"] = states.count("NO_CLAIM")
        doc_row["n_false_positive"] = sum(1 for r in rows if r["raw_terms"] and r["reason"] == "FALSE_POSITIVE")
        n_err = sum(1 for p in preds if "error" in p.details)
        doc_row["n_classifier_errors"] = n_err
        if n_err:
            log.warning("%s: %d sentences could not be classified (see 'error' in details)", f.filing_id, n_err)
        doc_row["status"] = "ok"
        return DocumentOutcome(rows, doc_row, recall_rows)

    def process_filing(
        self, filing: Filing, source, *, verify_input: bool = False, expected_sha256: str | None = None
    ) -> DocumentOutcome:
        """Fetch and analyse one filing; failures are recorded, not raised."""
        try:
            doc = (
                source.fetch_document_verified(filing, expected_sha256)
                if verify_input
                else source.fetch_document(filing)
            )
            return self.analyse(doc)
        except Exception as exc:
            log.warning("failed to process %s: %s", filing.filing_id, exc)
            drow = self._doc_row(filing)
            drow.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return DocumentOutcome([], drow)

    def run_documents(self, documents: Iterable[Document]) -> Results:
        documents = list(documents)
        ids = [doc.filing.filing_id for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate filing identifiers in documents; supply unique filings")
        manifest = _document_manifest(documents)
        outcomes = []
        for doc in documents:
            try:
                outcomes.append(self.analyse(doc))
            except Exception as exc:  # keep going; record the failure
                log.exception("failed to process %s", doc.filing.filing_id)
                drow = self._doc_row(doc.filing)
                drow.update(status="error", error=f"{type(exc).__name__}: {exc}")
                outcomes.append(DocumentOutcome([], drow))
        results = self._results(outcomes)
        results.provenance["input_manifest_sha256"] = manifest["sha256"]
        results.provenance["n_input_filings"] = manifest["n_filings"]
        results.provenance["source"] = {"name": "in-memory-documents", "manifest_schema": "document-v1"}
        return results

    def run(
        self,
        filings: Iterable[Filing],
        source,
        progress: bool = False,
        workers: int = 1,
        executor: str = "thread",
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
        retry_errors: bool = True,
    ) -> Results:
        """Fetch and process filings from a :class:`~aidisclose.sources.base.Source`.

        Parameters
        ----------
        workers, executor:
            Number of concurrent workers and ``"thread"`` or ``"process"``.
        checkpoint_dir:
            Directory for one checkpoint file per filing. Without ``resume``
            any existing checkpoints there are discarded first.
        resume:
            Reuse checkpoints written by an earlier run with the same
            configuration; filings that failed are retried unless
            ``retry_errors`` is false.
        """
        filings = list(filings)
        if checkpoint_dir and self.classifier.name == "transformer":
            raise ValueError(
                "transformer checkpointing is disabled until immutable model and tokenizer identities can be verified"
            )
        ids = [f.filing_id for f in filings]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate filing identifiers in input; supply unique filings")
        if executor not in {"thread", "process"}:
            raise ValueError("executor must be 'thread' or 'process'")
        manifest = _filing_manifest(filings, source)
        source_config = source.config() if hasattr(source, "config") else {"name": getattr(source, "name", "source")}
        can_verify = getattr(source, "can_verify_input_content", None)
        verify_input = bool(
            callable(can_verify) and can_verify() and callable(getattr(source, "fetch_document_verified", None))
        )
        if verify_input:
            source_config = dict(source_config, input_content_verified=True)
        content_hashes = manifest["local_content_hashes"]
        run_fingerprint = hashlib.sha256(
            (self.fingerprint(source_config) + ":" + manifest["sha256"]).encode("utf-8")
        ).hexdigest()
        ckpt = Checkpoint(checkpoint_dir, run_fingerprint, resume=resume) if checkpoint_dir else None
        done = ckpt.statuses() if ckpt else {}
        todo = [
            f
            for f in filings
            if f.filing_id not in done or (retry_errors and done[f.filing_id] in {"error", "classifier_error"})
        ]
        if ckpt and len(todo) < len(filings):
            log.info("resuming: %d of %d filings already processed", len(filings) - len(todo), len(filings))
        outcomes: dict[str, DocumentOutcome] = {}
        total = len(todo)

        def record(i: int, filing: Filing, out: DocumentOutcome) -> None:
            outcomes[filing.filing_id] = out
            if ckpt:
                ckpt.write(filing.filing_id, out)
            if progress:
                print(
                    f"[{i}/{total}] {filing.firm_name} {filing.fiscal_year} ({filing.filing_id}): "
                    f"{out.doc_row.get('status')}",
                    file=sys.stderr,
                )

        if workers <= 1 or total <= 1:
            for i, filing in enumerate(todo, 1):
                record(
                    i,
                    filing,
                    self.process_filing(
                        filing, source, verify_input=verify_input, expected_sha256=content_hashes.get(filing.filing_id)
                    ),
                )
        elif executor == "thread":
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self.process_filing,
                        f,
                        source,
                        verify_input=verify_input,
                        expected_sha256=content_hashes.get(f.filing_id),
                    ): f
                    for f in todo
                }
                for i, fut in enumerate(cf.as_completed(futures), 1):
                    record(i, futures[fut], fut.result())
        else:
            clients = _http_clients(source)
            # A single shared clock and lock prevents simultaneous first
            # requests from bypassing the rate limit in separate processes.
            ctx = mp.get_context()
            shared_rate = (
                (
                    ctx.Lock(),
                    ctx.Value("d", 0.0, lock=False),
                    max((client.min_interval for client in clients), default=0.0),
                )
                if clients
                else None
            )
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(self, source, workers, shared_rate),
            ) as pool:
                futures = {pool.submit(_work, f, verify_input, content_hashes.get(f.filing_id)): f for f in todo}
                for i, fut in enumerate(cf.as_completed(futures), 1):
                    filing = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as exc:  # e.g. a worker died
                        drow = self._doc_row(filing)
                        drow.update(status="error", error=f"{type(exc).__name__}: {exc}")
                        out = DocumentOutcome([], drow)
                    record(i, filing, out)
        ordered = []
        for f in filings:
            out = outcomes.get(f.filing_id)
            if out is None and ckpt:
                out = ckpt.read(f.filing_id)
                if out is None:
                    log.warning("checkpoint for %s disappeared or changed; reprocessing", f.filing_id)
                    out = self.process_filing(
                        f, source, verify_input=verify_input, expected_sha256=content_hashes.get(f.filing_id)
                    )
                    ckpt.write(f.filing_id, out)
            if out is not None:
                ordered.append(out)
        results = self._results(ordered)
        results.provenance["input_manifest_sha256"] = manifest["sha256"]
        results.provenance["n_input_filings"] = manifest["n_filings"]
        results.provenance["source"] = source_config
        return results

    # ----------------------------------------------------------- relabelling
    def candidates_from_results(self, results: Results) -> list[Candidate]:
        """Rebuild the classified candidates of an earlier run (rows with valid terms)."""
        out = []
        languages = (
            results.documents.set_index("filing_id")["language"].to_dict()
            if "language" in results.documents.columns
            else {}
        )
        for row in results.sentences.to_dict(orient="records"):
            terms = [t for t in str(row.get("terms") or "").split(";") if t]
            if not terms:
                continue
            text = str(row["text"])
            recorded = row.get("term_spans")
            if recorded is not None and str(recorded).strip() not in ("", "nan", "None", "<NA>"):
                try:
                    spans = json.loads(recorded) if isinstance(recorded, str) else recorded
                    if (
                        not isinstance(spans, list)
                        or not spans
                        or any(
                            not isinstance(s, (list, tuple))
                            or len(s) != 2
                            or any(isinstance(v, bool) or not isinstance(v, int) for v in s)
                            or not (0 <= s[0] < s[1] <= len(text))
                            for s in spans
                        )
                    ):
                        raise ValueError("invalid spans")
                    spans = [(s[0], s[1]) for s in spans]
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid recorded term spans for {row['sentence_id']}") from exc
            else:
                # Legacy outputs did not retain offsets. Reconstruct only
                # using a verifiably identical lexicon for this language.
                language = languages.get(row["filing_id"])
                if (language is None or pd.isna(language) or not str(language).strip()) and results.provenance.get(
                    "lexicons"
                ):
                    raise ValueError("legacy multilingual relabel needs each document's language")
                lang = (
                    "en"
                    if language is None or pd.isna(language) or not str(language).strip()
                    else str(language).lower()
                )
                expected = (
                    results.provenance.get("lexicon")
                    if lang == "en"
                    else results.provenance.get("lexicons", {}).get(lang)
                )
                actual = self.lexicons.get(lang)
                if actual is None or not isinstance(expected, dict):
                    raise ValueError(f"legacy relabel needs the original {lang} lexicon and provenance")
                desc = actual.describe()
                expected_terms = expected.get("terms_sha256")
                same = (
                    bool(expected_terms)
                    and expected_terms == desc["terms_sha256"]
                    and expected.get("version") == desc["version"]
                )
                if not expected_terms:
                    same = (
                        bool(expected.get("sha256"))
                        and expected.get("sha256") == desc["sha256"]
                        and expected.get("version") == desc["version"]
                    )
                if not same:
                    raise ValueError(f"legacy relabel needs the original {lang} lexicon with matching digest and tiers")
                hits = actual.find(text)
                wanted = set(terms)
                spans = [(h.start, h.end) for h in hits.valid if h.term_id in wanted]
                if {h.term_id for h in hits.valid if h.term_id in wanted} != wanted:
                    raise ValueError(f"cannot reconstruct all term spans for {row['sentence_id']}; rerun extraction")
            out.append(
                Candidate(
                    filing_id=str(row["filing_id"]),
                    sentence_id=str(row["sentence_id"]),
                    section=str(row["section"]),
                    text=text,
                    prev_text=str(row.get("prev_text") or ""),
                    next_text=str(row.get("next_text") or ""),
                    firm_name=str(row.get("firm_name") or ""),
                    terms=terms,
                    term_spans=spans,
                    categories=[c for c in str(row.get("categories") or "").split(";") if c],
                    raw_terms=[t for t in str(row.get("raw_terms") or "").split(";") if t],
                    filtered=[t for t in str(row.get("filtered_terms") or "").split(";") if t],
                    baseline_terms=[t for t in str(row.get("baseline_terms") or "").split(";") if t],
                )
            )
        return out

    def relabel(self, results: Results) -> Results:
        """Re-classify the candidate sentences of an earlier run with this pipeline's backend.

        Documents are not downloaded or segmented again: the sentences, their
        lexicon terms and the document rows are reused, and only the state,
        reason, confidence, backend and details columns (and the per-document
        counts) change. Use it to compare backends on identical candidates.
        """
        cands = self.candidates_from_results(results)
        preds = self.classifier.classify(cands) if cands else []
        if len(preds) != len(cands):
            raise ValueError(f"classifier returned {len(preds)} predictions for {len(cands)} candidates")
        if len({c.sentence_id for c in cands}) != len(cands):
            raise ValueError("duplicate sentence identifiers in relabel candidates")
        new = {c.sentence_id: p.as_row() for c, p in zip(cands, preds)}
        sentences = results.sentences.copy()
        for col in ("state", "reason", "classification_status", "confidence", "backend", "details"):
            sentences[col] = [
                new[sid][col] if sid in new else old for sid, old in zip(sentences["sentence_id"], sentences[col])
            ]
        sentences["confidence"] = pd.to_numeric(sentences["confidence"], errors="coerce")
        documents = results.documents.copy()
        for i, fid in enumerate(documents["filing_id"]):
            grp = sentences[(sentences["filing_id"] == fid) & (sentences["raw_terms"].astype(str) != "")]
            states = [s for s in grp["state"].tolist() if s in {v.value for v in AdoptionState}]
            documents.loc[documents.index[i], "n_substantive"] = states.count("SUBSTANTIVE")
            documents.loc[documents.index[i], "n_weak"] = states.count("WEAK")
            documents.loc[documents.index[i], "n_no_claim"] = states.count("NO_CLAIM")
            documents.loc[documents.index[i], "n_false_positive"] = int((grp["reason"] == "FALSE_POSITIVE").sum())
            documents.loc[documents.index[i], "n_classifier_errors"] = int((grp["classification_status"] != "ok").sum())
        prov = dict(results.provenance)
        prov["classifier"] = self.classifier.config()
        prov["relabelled_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        prov["relabelled_with_version"] = __version__
        prov["relabelled_from_classifier"] = results.provenance.get("classifier")
        return Results(sentences=sentences, documents=documents, provenance=prov, recall_pool=results.recall_pool)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _doc_row(f: Filing) -> dict[str, Any]:
        row = {c: "" for c in DOCUMENT_COLUMNS}
        row.update(
            filing_id=f.filing_id,
            firm_id=f.firm_id,
            firm_name=f.firm_name,
            fiscal_year=f.fiscal_year,
            period_end=f.period_end or "",
            country=f.country or "",
            industry_code=f.industry_code or "",
            industry_scheme=f.industry_scheme or "",
            source=f.source,
            form=f.form or "",
            url=f.url or "",
            language=f.language or "",
            extra=json.dumps(f.extra, sort_keys=True, default=str),
        )
        for c in DOCUMENT_COLUMNS:
            if c.startswith("n_"):
                row[c] = 0
        return row

    def _results(self, outcomes: Sequence[DocumentOutcome]) -> Results:
        sent_rows = [r for o in outcomes for r in o.rows]
        doc_rows = [o.doc_row for o in outcomes]
        pool_rows = [r for o in outcomes for r in o.recall_rows]
        sentences = pd.DataFrame(sent_rows, columns=SENTENCE_COLUMNS)
        documents = pd.DataFrame(doc_rows, columns=DOCUMENT_COLUMNS)
        pool = pd.DataFrame(pool_rows, columns=RECALL_COLUMNS)
        for df in (sentences, documents, pool):
            df["fiscal_year"] = pd.array([_int_or_none(v) for v in df["fiscal_year"]], dtype="Int64")
        return Results(sentences=sentences, documents=documents, provenance=self.provenance(), recall_pool=pool)


def _int_or_none(v) -> int | None:
    if v is None or v is pd.NA or v == "" or v != v:
        return None
    return int(v)


def _distribution_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _document_manifest(documents: Sequence[Document]) -> dict[str, Any]:
    """Hash ordered in-memory inputs, including text, ordered sections and declared language."""
    rows = []
    for doc in documents:
        rows.append(
            {
                "filing": doc.filing.to_dict(),
                "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                "sections": [
                    [name, hashlib.sha256(text.encode("utf-8")).hexdigest()] for name, text in doc.sections.items()
                ],
                "language": doc.language,
            }
        )
    encoded = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return {"n_filings": len(documents), "sha256": hashlib.sha256(encoded).hexdigest()}


def _filing_manifest(filings: Sequence[Filing], source=None) -> dict[str, Any]:
    """Hash ordered filing metadata and local bytes available at run start."""
    rows = []
    content_hashes = {}
    for filing in filings:
        row = filing.to_dict()
        resolve_path = getattr(source, "input_content_path", None)
        local_path = resolve_path(filing) if callable(resolve_path) else filing.extra.get("path")
        if local_path and Path(local_path).is_file():
            digest = hashlib.sha256()
            with Path(local_path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            row["local_content_sha256"] = digest.hexdigest()
            content_hashes[filing.filing_id] = row["local_content_sha256"]
        elif callable(resolve_path):
            row["local_content_state"] = "unavailable"
            content_hashes[filing.filing_id] = None
        rows.append(row)
    encoded = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return {
        "n_filings": len(filings),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "local_content_hashes": content_hashes,
    }


def _context(spans, i: int) -> tuple[str, str]:
    sp = spans[i]
    prev_text = spans[i - 1].text if i > 0 and spans[i - 1].block >= sp.block - 2 else ""
    next_text = spans[i + 1].text if i + 1 < len(spans) and spans[i + 1].block <= sp.block + 2 else ""
    return prev_text, next_text


# ------------------------------------------------------------------ checkpoints
class Checkpoint:
    """One JSON file per processed filing, plus the configuration fingerprint."""

    def __init__(self, directory: str | Path, fingerprint: str, resume: bool = False):
        self.dir = Path(directory)
        self.parts = self.dir / "parts"
        meta_path = self.dir / "checkpoint.json"
        if resume and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"cannot resume from {self.dir}: it was written with a different configuration "
                    "(lexicon, backend, sections or languages); use a new directory or run without resume"
                )
        elif resume and self.parts.exists() and any(self.parts.glob("*.json")):
            raise ValueError(
                "cannot resume: checkpoint metadata is missing; preserve the parts and start a new directory"
            )
        elif not resume and self.parts.exists():
            for p in self.parts.glob("*.json"):
                p.unlink()
        self.parts.mkdir(parents=True, exist_ok=True)
        tmp_meta = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.dir, prefix=".checkpoint-", encoding="utf-8", delete=False
            ) as stream:
                tmp_meta = Path(stream.name)
                stream.write(json.dumps({"fingerprint": fingerprint, "aidisclose_version": __version__}, indent=2))
            os.replace(tmp_meta, meta_path)
        finally:
            if tmp_meta is not None:
                tmp_meta.unlink(missing_ok=True)

    def _path(self, filing_id: str) -> Path:
        return self.parts / (hashlib.sha1(filing_id.encode("utf-8")).hexdigest()[:20] + ".json")

    def write(self, filing_id: str, out: DocumentOutcome) -> None:
        payload = {"filing_id": filing_id, "rows": out.rows, "doc_row": out.doc_row, "recall_rows": out.recall_rows}
        payload["part_sha256"] = self._part_digest(payload)
        path = self._path(filing_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _part_digest(payload: dict) -> str:
        data = {k: v for k, v in payload.items() if k != "part_sha256"}
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _valid_data(self, path: Path, expected_id: str | None = None) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            checksum = data.get("part_sha256")
            if not isinstance(checksum, str) or not hmac.compare_digest(checksum, self._part_digest(data)):
                return None
            fid = data["filing_id"]
            doc = data["doc_row"]
            rows, recall = data["rows"], data.get("recall_rows", [])
            if not isinstance(fid, str) or not fid or path != self._path(fid):
                return None
            if expected_id is not None and fid != expected_id:
                return None
            if not isinstance(doc, dict) or doc.get("filing_id") != fid:
                return None
            if doc.get("status") not in {
                "ok",
                "error",
                "skipped_non_english",
                "skipped_language",
                "skipped_no_lexicon",
            }:
                return None
            if not isinstance(rows, list) or not isinstance(recall, list):
                return None
            if any(not isinstance(row, dict) or row.get("filing_id") != fid for row in rows + recall):
                return None
            int(doc.get("n_classifier_errors", 0) or 0)
            return data
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def read(self, filing_id: str) -> DocumentOutcome | None:
        data = self._valid_data(self._path(filing_id), filing_id)
        if data is None:
            return None
        return DocumentOutcome(data["rows"], data["doc_row"], data.get("recall_rows", []))

    def statuses(self) -> dict[str, str]:
        out = {}
        for p in self.parts.glob("*.json"):
            data = self._valid_data(p)
            if data is None:
                log.warning("ignoring invalid checkpoint part %s; filing will be processed again", p.name)
                continue
            out[data["filing_id"]] = str(data["doc_row"].get("status", ""))
            if int(data["doc_row"].get("n_classifier_errors", 0) or 0) > 0:
                out[data["filing_id"]] = "classifier_error"
        return out


def _json_default(obj):
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if obj is pd.NA:
        return None
    return str(obj)


# ------------------------------------------------------------- process workers
_WORKER: dict[str, Any] = {}


def _init_worker(pipeline: Pipeline, source, n_workers: int, shared_rate=None) -> None:  # pragma: no cover
    for http in _http_clients(source):
        http._shared_rate = shared_rate
    _WORKER["pipeline"], _WORKER["source"] = pipeline, source


def _work(
    filing: Filing, verify_input: bool = False, expected_sha256: str | None = None
) -> DocumentOutcome:  # pragma: no cover
    return _WORKER["pipeline"].process_filing(
        filing, _WORKER["source"], verify_input=verify_input, expected_sha256=expected_sha256
    )


def _http_clients(obj) -> list:
    from .sources.base import HttpClient

    return [v for v in getattr(obj, "__dict__", {}).values() if isinstance(v, HttpClient)]
