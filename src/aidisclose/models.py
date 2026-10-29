"""Core data structures shared by all AIDisclose modules.

The measurement unit of AIDisclose is the *candidate sentence*: a sentence of a
corporate disclosure that contains at least one term of the AI lexicon. Every
candidate receives exactly one :class:`AdoptionState` and one :class:`Reason`.
Document-, firm- and group-level indicators are aggregations of these labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


class AdoptionState(str, Enum):
    """Three-state taxonomy for AI-related statements in corporate text.

    SUBSTANTIVE
        The focal (reporting) firm states that it currently or previously
        uses, deploys, sells or has invested in a concrete AI system,
        product or capability, with at least one specific element
        (application, product, quantity or investment).
    WEAK
        The focal firm refers to AI in a hedged, forward-looking,
        exploratory, risk-boilerplate or unspecific way.
    NO_CLAIM
        The sentence contains no valid focal-firm claim about AI adoption:
        it is about third parties, markets or regulation, it negates
        adoption, or the keyword is a false positive.
    """

    SUBSTANTIVE = "SUBSTANTIVE"
    WEAK = "WEAK"
    NO_CLAIM = "NO_CLAIM"

    @property
    def rank(self) -> int:
        return STATE_RANK[self]


STATE_RANK = {
    AdoptionState.NO_CLAIM: 0,
    AdoptionState.WEAK: 1,
    AdoptionState.SUBSTANTIVE: 2,
}

STATES = [s.value for s in AdoptionState]


class Reason(str, Enum):
    """Reason codes that explain why a state was assigned."""

    # SUBSTANTIVE
    DEPLOYMENT = "DEPLOYMENT"  # concrete current/past use or deployment
    PRODUCT = "PRODUCT"  # AI-enabled product or service offered by the firm
    INVESTMENT = "INVESTMENT"  # acquisition, patents, dedicated teams, compute
    # WEAK
    FORWARD_LOOKING = "FORWARD_LOOKING"  # plans, intentions, pilots, exploration
    HEDGED = "HEDGED"  # modal or conditional statements
    RISK_BOILERPLATE = "RISK_BOILERPLATE"  # generic risk-factor language
    VAGUE = "VAGUE"  # focal claim without any specific element
    # NO_CLAIM
    FALSE_POSITIVE = "FALSE_POSITIVE"  # keyword used in a non-AI sense
    THIRD_PARTY = "THIRD_PARTY"  # competitors, vendors, customers, attackers
    MARKET_TREND = "MARKET_TREND"  # industry or economy-wide statements
    REGULATORY = "REGULATORY"  # laws, regulation and policy about AI
    NEGATED = "NEGATED"  # explicit statement of non-adoption
    GENERIC = "GENERIC"  # no identifiable agent
    # any state, used by model backends that only predict the state
    UNSPECIFIED = "UNSPECIFIED"


REASONS_BY_STATE: dict[AdoptionState, tuple[Reason, ...]] = {
    AdoptionState.SUBSTANTIVE: (Reason.DEPLOYMENT, Reason.PRODUCT, Reason.INVESTMENT),
    AdoptionState.WEAK: (
        Reason.FORWARD_LOOKING,
        Reason.HEDGED,
        Reason.RISK_BOILERPLATE,
        Reason.VAGUE,
    ),
    AdoptionState.NO_CLAIM: (
        Reason.FALSE_POSITIVE,
        Reason.THIRD_PARTY,
        Reason.MARKET_TREND,
        Reason.REGULATORY,
        Reason.NEGATED,
        Reason.GENERIC,
    ),
}


def coerce_state(value: Any) -> AdoptionState:
    """Return an :class:`AdoptionState` from a string or enum (case-insensitive)."""
    if isinstance(value, AdoptionState):
        return value
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {"A1": "SUBSTANTIVE", "A2": "WEAK", "A3": "NO_CLAIM", "NONE": "NO_CLAIM"}
    text = aliases.get(text, text)
    return AdoptionState(text)


def coerce_reason(value: Any, state: AdoptionState | None = None) -> Reason:
    """Return a :class:`Reason`; fall back to UNSPECIFIED when invalid for *state*."""
    try:
        reason = value if isinstance(value, Reason) else Reason(str(value).strip().upper())
    except ValueError:
        return Reason.UNSPECIFIED
    if state is not None and reason is not Reason.UNSPECIFIED:
        if reason not in REASONS_BY_STATE[state]:
            return Reason.UNSPECIFIED
    return reason


def _validated_fiscal_year(value: Any) -> int | None:
    """Accept missing years or finite integral Gregorian calendar years only."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "<na>", "none"}:
        return None
    if isinstance(value, bool):
        raise ValueError("fiscal_year must be an integer calendar year or missing")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid fiscal_year {value!r}; expected an integer calendar year") from exc
    if not number.is_finite() or number != number.to_integral_value() or not 1 <= number <= 9999:
        raise ValueError(f"invalid fiscal_year {value!r}; expected an integer calendar year between 1 and 9999")
    return int(number)


@dataclass
class Filing:
    """Metadata describing one disclosure document, independent of its source."""

    source: str
    filing_id: str
    firm_id: str
    firm_name: str
    fiscal_year: int | None = None
    period_end: str | None = None  # ISO date of the fiscal period end
    country: str | None = None  # ISO 3166-1 alpha-2
    industry_code: str | None = None
    industry_scheme: str | None = None  # e.g. "SIC", "NACE"
    form: str | None = None  # e.g. "10-K", "ESEF"
    url: str | None = None
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fiscal_year = _validated_fiscal_year(self.fiscal_year)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["extra"] = json.dumps(self.extra, sort_keys=True, default=str)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filing:
        data = dict(data)
        extra = data.get("extra") or {}
        if isinstance(extra, str):
            extra = json.loads(extra) if extra.strip() else {}
        data["extra"] = extra
        known = {k: data.get(k) for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        known["fiscal_year"] = _validated_fiscal_year(known.get("fiscal_year"))
        for key, value in list(known.items()):
            if (
                key not in ("extra", "fiscal_year")
                and value is not None
                and not isinstance(value, str)
                and str(value).lower() in {"nan", "<na>"}
            ):
                known[key] = None
        return cls(**known)


@dataclass
class Document:
    """Extracted plain text of a filing, optionally split into sections."""

    filing: Filing
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    language: str | None = None
    language_scores: dict[str, float] = field(default_factory=dict)

    def iter_sections(self):
        if self.sections:
            yield from self.sections.items()
        else:
            yield "full", self.text

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Candidate:
    """A sentence that contains at least one AI lexicon term."""

    filing_id: str
    sentence_id: str
    section: str
    text: str
    prev_text: str = ""
    next_text: str = ""
    firm_name: str = ""
    terms: list[str] = field(default_factory=list)  # valid (retained) term ids
    term_spans: list[tuple[int, int]] = field(default_factory=list)  # offsets of valid hits
    categories: list[str] = field(default_factory=list)
    raw_terms: list[str] = field(default_factory=list)  # all hits before filtering
    filtered: list[str] = field(default_factory=list)  # hits removed as ambiguous
    filter_reasons: dict[str, str] = field(default_factory=dict)  # term id -> why it was removed
    baseline_terms: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """Label assigned to a candidate by a classifier backend."""

    state: AdoptionState
    reason: Reason
    confidence: float = 1.0
    backend: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        status = "error" if "error" in self.details else "ok"
        return {
            # A backend failure is missing information, not a substantive
            # conclusion that the sentence contains no focal-firm claim.
            "state": self.state.value if status == "ok" else "",
            "reason": self.reason.value if status == "ok" else "",
            "classification_status": status,
            "confidence": round(float(self.confidence), 4),
            "backend": self.backend,
            "details": json.dumps(self.details, sort_keys=True, default=str),
        }
