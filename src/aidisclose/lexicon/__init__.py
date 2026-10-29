"""Versioned, editable AI lexicons with disambiguation rules.

A lexicon is a YAML file listing regular-expression patterns for AI terms (see
``lexicon/data/aidisclose-en.yaml`` for the documented format). Lexicons are
identified by name and version, and every result produced by AIDisclose records
the SHA-256 digest of the lexicon file it used, so measurements can be traced to
an exact term list.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import yaml

BUILTIN = {
    "default": "aidisclose-en.yaml",
    "aidisclose-en": "aidisclose-en.yaml",
    # Frozen lexicon for exact replication of published results.
    "aidisclose-en-1.0.0": "aidisclose-en-1.0.0.yaml",
    "keyword-baseline-4": "keyword-baseline-4.yaml",
    # stratification aid for the recall audit; not a measurement lexicon
    "recall-probe": "recall-probe-en.yaml",
}

_LEFT = r"(?<![A-Za-z0-9])"
_RIGHT = r"(?![A-Za-z0-9])"


@dataclass(frozen=True)
class Term:
    id: str
    pattern: str
    category: str
    tier: str = "core"
    case_sensitive: bool = False
    ambiguous: bool = False
    requires_cooccurrence: bool = False
    negative_contexts: tuple[str, ...] = ()
    added: str = ""

    def compile(self) -> re.Pattern[str]:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(_LEFT + "(?:" + self.pattern + ")" + _RIGHT, flags)


@dataclass(frozen=True)
class Hit:
    term_id: str
    category: str
    tier: str
    start: int
    end: int
    surface: str
    ambiguous: bool = False


@dataclass
class LexiconHits:
    raw: list[Hit] = field(default_factory=list)  # every pattern hit
    valid: list[Hit] = field(default_factory=list)  # retained after disambiguation
    filtered: list[Hit] = field(default_factory=list)  # removed as false positives
    reasons: dict[str, str] = field(default_factory=dict)  # term id -> why it was removed

    @property
    def any_raw(self) -> bool:
        return bool(self.raw)

    def term_ids(self, which: str = "valid") -> list[str]:
        return sorted({h.term_id for h in getattr(self, which)})

    def categories(self) -> list[str]:
        return sorted({h.category for h in self.valid})


class Lexicon:
    """A compiled set of AI terms.

    Parameters
    ----------
    name, version, language:
        Identification metadata copied from the YAML file.
    terms:
        Terms to match.
    sha256:
        Digest of the source file (empty for programmatically built lexicons).
    """

    def __init__(
        self,
        name: str,
        version: str,
        terms: Iterable[Term],
        language: str = "en",
        description: str = "",
        sha256: str = "",
        source: str = "",
    ) -> None:
        self.name = name
        self.version = version
        self.language = language
        self.description = description
        self.sha256 = sha256
        self.source = source
        self.terms = list(terms)
        ids = [t.id for t in self.terms]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate term ids in lexicon {name!r}")
        self._compiled = [(t, t.compile()) for t in self.terms]
        self._negatives = {t.id: [re.compile(p, re.IGNORECASE) for p in t.negative_contexts] for t in self.terms}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, name_or_path: str | Path = "default", tiers: Iterable[str] | None = ("core",)) -> Lexicon:
        """Load a built-in lexicon by name or a YAML file by path.

        ``tiers`` restricts the terms to the given tiers (``None`` keeps all).
        """
        key = str(name_or_path)
        if key in BUILTIN:
            ref = resources.files("aidisclose.lexicon").joinpath("data", BUILTIN[key])
            raw = ref.read_bytes()
            source = "builtin:" + BUILTIN[key]
        else:
            path = Path(name_or_path)
            if not path.exists():
                raise FileNotFoundError(f"lexicon {key!r} is neither a built-in name ({sorted(BUILTIN)}) nor a file")
            raw = path.read_bytes()
            source = str(path)
        return cls.from_yaml_bytes(raw, tiers=tiers, source=source)

    @classmethod
    def from_yaml_bytes(cls, raw: bytes, tiers: Iterable[str] | None = ("core",), source: str = "") -> Lexicon:
        data = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(data, dict) or "terms" not in data:
            raise ValueError("lexicon YAML must be a mapping with a 'terms' list")
        tier_set = None if tiers is None else set(tiers)
        terms = []
        for item in data["terms"]:
            missing = {"id", "pattern", "category"} - set(item)
            if missing:
                raise ValueError(f"lexicon term {item!r} is missing {sorted(missing)}")
            term = Term(
                id=str(item["id"]),
                pattern=str(item["pattern"]),
                category=str(item["category"]),
                tier=str(item.get("tier", "core")),
                case_sensitive=bool(item.get("case_sensitive", False)),
                ambiguous=bool(item.get("ambiguous", False)),
                requires_cooccurrence=bool(item.get("requires_cooccurrence", False)),
                negative_contexts=tuple(item.get("negative_contexts") or ()),
                added=str(item.get("added", "")),
            )
            if tier_set is None or term.tier in tier_set:
                terms.append(term)
        tier_label = "all" if tier_set is None else "+".join(sorted(tier_set))
        return cls(
            name=str(data.get("name", "custom")),
            version=f"{data.get('version', '0')}[{tier_label}]",
            terms=terms,
            language=str(data.get("language", "en")),
            description=str(data.get("description", "")),
            sha256=hashlib.sha256(raw).hexdigest(),
            source=source,
        )

    # ----------------------------------------------------------------- matching
    def find(self, sentence: str, disabled_terms: Iterable[str] = ()) -> LexiconHits:
        """Match all terms in *sentence* and apply disambiguation rules.

        ``disabled_terms`` lists term ids that must be treated as false
        positives for this sentence (used for document-level acronym
        definitions such as "Artificial Insemination (AI)").
        """
        disabled = set(disabled_terms)
        hits = LexiconHits()
        for term, rx in self._compiled:
            for m in rx.finditer(sentence):
                hits.raw.append(Hit(term.id, term.category, term.tier, m.start(), m.end(), m.group(0), term.ambiguous))
        if not hits.raw:
            return hits
        by_id = {t.id: t for t in self.terms}
        unambiguous_present = any(not by_id[h.term_id].ambiguous and h.term_id not in disabled for h in hits.raw)
        # pass 1: every term that does not need a co-occurring AI term
        decided: dict[int, str | None] = {}  # index -> rejection reason (None = kept)
        for i, h in enumerate(hits.raw):
            term = by_id[h.term_id]
            if term.requires_cooccurrence:
                continue
            reason = "document_definition" if h.term_id in disabled else None
            if reason is None and term.negative_contexts and not unambiguous_present:
                if any(rx.search(sentence) for rx in self._negatives[term.id]):
                    reason = "negative_context"
            decided[i] = reason
        # pass 2: acronyms such as "ML" are kept only next to another valid AI term ("AI/ML")
        other_valid = any(r is None for r in decided.values())
        for i, h in enumerate(hits.raw):
            if i not in decided:
                term = by_id[h.term_id]
                if h.term_id in disabled:
                    reason = "document_definition"
                elif not other_valid:
                    reason = "requires_cooccurrence"
                elif (
                    term.negative_contexts
                    and not unambiguous_present
                    and any(rx.search(sentence) for rx in self._negatives[term.id])
                ):
                    reason = "negative_context"
                else:
                    reason = None
                decided[i] = reason
            if decided[i] is None:
                hits.valid.append(h)
            else:
                hits.filtered.append(h)
                hits.reasons.setdefault(h.term_id, decided[i])
        return hits

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Lexicon(name={self.name!r}, version={self.version!r}, terms={len(self.terms)})"

    def describe(self) -> dict:
        # YAML file hashes alone cannot identify programmatically constructed
        # lexicons, nor the active tier subset of a YAML file.
        canonical = json.dumps(
            [asdict(term) for term, _ in self._compiled], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return {
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "n_terms": len(self.terms),
            "sha256": self.sha256,
            "terms_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "source": self.source,
        }


# --------------------------------------------------------------------------
# Document-level acronym definitions, e.g. 'Artificial Insemination ("AI")'.
_DEF_RX = re.compile(r"((?:[A-Za-z][\w'-]*\s+){1,4})\(\s*(?:the\s+)?[\"“”']?(AI|ML|NLP|LLM)s?[\"“”']?\s*\)")
_EXPANSIONS = {
    "AI": re.compile(r"artificial[\s-]+intelligence", re.I),
    "ML": re.compile(r"machine[\s-]+learning", re.I),
    "NLP": re.compile(r"natural[\s-]+language", re.I),
    "LLM": re.compile(r"language[\s-]+models?", re.I),
}
_ACRONYM_TERM = {"AI": "ai_acronym", "ML": "ml_acronym", "NLP": "nlp_acronym", "LLM": "llm_acronym"}


def conflicting_acronyms(text: str) -> set[str]:
    """Return lexicon term ids whose acronym the document defines as a non-AI concept.

    Example: a filing that writes 'artificial insemination ("AI")' and never
    defines AI as artificial intelligence returns ``{"ai_acronym"}``.
    """
    ai_defs: dict[str, bool] = {}
    for m in _DEF_RX.finditer(text):
        words, acro = m.group(1), m.group(2)
        is_ai = bool(_EXPANSIONS[acro].search(words))
        if is_ai:
            ai_defs[acro] = True
        else:
            initials = "".join(w[0].upper() for w in words.split() if w[:1].isalpha())
            if initials.endswith(acro) and acro not in ai_defs:
                ai_defs[acro] = False
    return {_ACRONYM_TERM[a] for a, ok in ai_defs.items() if not ok}


__all__ = ["Lexicon", "Term", "Hit", "LexiconHits", "conflicting_acronyms", "BUILTIN"]
