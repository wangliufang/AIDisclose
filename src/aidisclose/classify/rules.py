"""Transparent rule-based classifier (default backend).

The rules implement the decision procedure of the AIDisclose codebook
(``annotation/codebook.md``). For every valid AI term in a sentence the
classifier

1. identifies the *agent* of the AI mention -- the focal (reporting) firm, a
   third party, or nobody -- as the nearest agent cue preceding the term
   (falling back to the first cue after it, and, for passive constructions,
   to the previous sentence),
2. screens the agent's clause for negation, and the term's neighbourhood for
   regulatory and market-trend context,
3. looks for deployment, product and investment cues tied to the focal firm,
   and for forward-looking, hedging and risk cues that weaken the claim, and
4. requires at least one *specific* element (application, quantity, date,
   named product or investment) before a focal claim is labelled SUBSTANTIVE.

The sentence label is the strongest label over its AI terms. All cues that
fired are returned in ``Prediction.details`` so that every label can be
audited. The rules need no model download and are deterministic; use the
validation kit to measure their accuracy on your corpus and switch to the
``supervised`` or ``llm`` backends where they fall short.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import AdoptionState, Candidate, Prediction, Reason
from ..text.sections import is_risk_section
from .base import Classifier

_I = re.IGNORECASE

FOCAL = re.compile(
    r"\b(?:[Ww]e|WE|[Oo]ur|OUR|[Oo]urs|us|[Oo]urselves)\b"
    r"|\b[Tt]he\s+(?:[Cc]ompany|[Gg]roup|[Cc]orporation|[Ff]irm|[Bb]ank|[Rr]egistrant|[Pp]artnership|[Tt]rust)(?:'s)?\b"
)
THIRD_PARTY = re.compile(
    r"\b(?:competitors?'?|competition|rivals?|third[\s-]+part(?:y|ies)|vendors?|suppliers?|"
    r"service\s+providers?|counterpart(?:y|ies)|"
    r"other\s+(?:companies|firms|banks|insurers|institutions|market\s+participants|businesses|parties|organi[sz]ations)|"
    r"new\s+entrants|start-?ups?|insurtechs?|fintechs?|hyperscalers?|(?:large\s+)?(?:technology|tech)\s+(?:companies|firms|giants)|"
    r"(?:customers?|clients?)'?(?!\s+(?:service|support|care|experience|interactions?|inquiries|engagement|journey|"
    r"onboarding|data|success|relationships?|retention|acquisition|insights?))|consumers|governments?|regulators?|"
    r"lawmakers|policymakers|supervisors|"
    r"threat\s+actors?|hackers?|cyber-?\s?criminals?|bad\s+actors?|malicious\s+actors?|fraudsters?|attackers?|adversaries|"
    r"many\s+(?:companies|businesses|organi[sz]ations)|employers|investors)\b",
    _I,
)
# an AI term that is part of the name of a law or regulator ("EU AI Act", "AI regulation")
REG_NAME_AFTER = re.compile(
    r"^[\s-]*(?:Act|Regulation|Directive|Bill|Law|Office|Pact|Convention|Code\s+of\s+(?:Practice|Conduct)|"
    r"regulations?|legislation|laws|rules)\b"
)
# "third-party AI models", "vendor-provided machine learning": supplied by others but used by the focal firm
ATTRIBUTIVE_TP = re.compile(
    r"\b(?:third[\s-]+part(?:y|ies)'?|vendors?'?|suppliers?'?|partners'|external|open[\s-]+source)"
    r"(?:[\s-]+(?:provided|supplied|developed|built|hosted))?\s+(?:[\w-]+\s+){0,2}$",
    _I,
)
REGULATOR_AGENT = re.compile(r"^(?:governments?|regulators?|lawmakers|policymakers|supervisors)$", _I)
SERVICE_TO_THIRD_PARTY = re.compile(
    r"\b(?:we|our)\b[^.;]{0,60}?\b(?:help|helps|helped|helping|enable|enables|enabled|enabling|assist|assists|assisted|"
    r"support|supports|supported|allow|allows|empower|empowers|advise|advises|equip|equips)\b[^.;]{0,30}$",
    _I,
)
MARKET = re.compile(
    r"\b(?:the\s+(?:industry|market|economy|sector|world)|our\s+(?:industry|industries|markets?|sector)|industries|"
    r"across\s+(?:the\s+)?(?:industry|economy|sectors?|markets?)|"
    r"(?:is|are)\s+(?:rapidly\s+)?(?:transforming|reshaping|disrupting|changing|evolving|revolutioni[sz]ing)|"
    r"(?:will|could|may|might|is\s+(?:expected|likely|set)\s+to|are\s+(?:expected|likely|set)\s+to|continues?\s+to)\s+"
    r"(?:fundamentally\s+|significantly\s+)?(?:transform|reshape|disrupt|revolutioni[sz]e|change|redefine)|"
    r"over\s+the\s+next\s+(?:decade|few\s+years)|"
    r"trends?|widespread|proliferation|growing\s+(?:use|adoption|interest)|increasing(?:ly)?\s+(?:adopt\w*|use|used|prevalent|common)|"
    r"(?:adoption|use)\s+of\s+\w+(?:\s+\w+)?\s+(?:by|across|throughout)\s+(?:businesses|companies|industr\w+|econom\w+|society))",
    _I,
)
REGULATORY = re.compile(
    r"\b(?:regulat\w*|legislat\w*|laws?|legal\s+(?:frameworks?|requirements?|regimes?)|AI\s+Act|"
    r"Artificial\s+Intelligence\s+Act|executive\s+orders?|policymakers|lawmakers|"
    r"rules?\s+(?:on|governing|relating\s+to|regarding)|governing\s+(?:the\s+)?(?:use|development|deployment)|"
    r"supervisory\s+(?:authorities|expectations|guidance)|data\s+protection\s+authorit\w*)\b",
    _I,
)
NEGATION = re.compile(
    r"\b(?:(?:do|does|did)\s+not\s+(?:currently\s+|yet\s+)?(?:use|utili[sz]e|employ|deploy|rely\s+on|develop|offer|have)|"
    r"ha(?:ve|s)\s+not\s+(?:yet\s+)?(?:used|deployed|adopted|implemented|developed|incorporated|integrated|introduced)|"
    r"(?:no|not\s+any)\s+(?:current\s+)?(?:use|deployment|adoption|application)\s+of|"
    r"not\s+(?:currently\s+)?(?:using|deploying|adopting|developing)|never\s+(?:used|deployed|adopted))\b",
    _I,
)
DEPLOY = re.compile(
    r"\b(?:use[sd]?|using|utili[sz](?:e|es|ed|ing)|employ(?:s|ed|ing)?|deploy(?:s|ed|ing|ment)?|"
    r"leverag(?:e|es|ed|ing)|appl(?:y|ies|ied|ying)|implement(?:s|ed|ing)?|integrat(?:e|es|ed|ing)|"
    r"incorporat(?:e|es|ed|ing)|embed(?:s|ded|ding)?|rolled\s+out|roll(?:s|ing)?\s+out|"
    r"automat(?:e|es|ed|ing)|train(?:s|ed|ing)?|power(?:s|ed|ing)?|run(?:s|ning)?|"
    r"operat(?:e|es|ed|ing)|rel(?:y|ies|ied|ying)\s+on|adopt(?:s|ed|ing)?|built|build(?:s|ing)?|"
    r"develop(?:s|ed|ing)?|designed|created|enhanced|improved|reduced|increased|generated|generates|processed|handled|"
    r"plans|planned|ranks|ranked|prices|priced|screened|screens|prioriti[sz](?:e|es|ed)|"
    r"detect(?:s|ed)|analy[sz](?:es|ed)|review(?:s|ed)|draft(?:s|ed))\b",
    _I,
)
OFFER = re.compile(
    r"\b(?:offer(?:s|ed|ing)?|sell(?:s|ing)?|sold|launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|"
    r"provid(?:e|es|ed|ing)|releas(?:e|es|ed|ing)|licens(?:e|es|ed|ing)|commerciali[sz](?:e|es|ed|ing)|ship(?:s|ped)?)\b",
    _I,
)
INVEST = re.compile(
    r"\b(?:invest(?:s|ed|ing|ments?)?|acqui(?:re|res|red|ring|sition|sitions)|patents?|patent\s+applications|"
    r"data\s+scientists|data\s+science\s+team|(?:machine\s+learning|ML|AI)\s+(?:engineers|researchers|specialists|team|teams)|"
    r"(?:research|innovation)\s+(?:lab|labs|laborator(?:y|ies)|cent(?:er|re)s?|team|teams)|R&D|"
    r"cent(?:er|re)s?\s+of\s+excellence|GPUs?|compute\s+capacity|hired|recruited)\b",
)
# investment *acts* (as opposed to mentions of teams or labs) decide the INVESTMENT reason
INVEST_ACT = re.compile(
    r"^(?:invest\w*|acqui\w+|patents?|patent\s+applications|GPUs?|compute\s+capacity|hired|recruited)$", _I
)
MODAL_AFTER = re.compile(r"^[^.;]{0,30}?\b(?:may|might|could|would|can)\b", _I)
PRODUCT_ADJ = re.compile(
    r"\b(?:AI|A\.I\.|artificial\s+intelligence|machine[\s-]+learning|ML|generative\s+AI|GenAI|LLM|deep[\s-]+learning)"
    r"[\s-]+(?:powered|driven|enabled|based|native|enhanced|assisted|infused|first|optimi[sz]ed)\b",
    _I,
)
PRODUCT_NOUN = re.compile(
    r"\b(?:products?|features?|apps?|applications?|software|services?|assistants?|copilots?|chatbots?|offerings?|"
    r"solutions?|platforms?|tools?|subscriptions?|modules?|accelerators?|chips?)\b",
    _I,
)
EXTERNAL_USER = re.compile(
    r"\b(?:customers?|clients?|users|subscribers|policyholders|patients|consumers|shoppers|members|businesses|"
    r"partners|developers)\b",
    _I,
)
USE_NOUN = re.compile(
    r"\b(?:[Oo]ur|[Tt]he\s+[Cc]ompany's)\s+(?:\w+\s+){0,2}?(?:use|usage|deployment|adoption|application|development|integration)\s+of\b"
)
POSSESSIVE_BEFORE = re.compile(r"\b(?:[Oo]ur|[Tt]he\s+[Cc]ompany's|[Tt]he\s+[Gg]roup's)\s+(?:[\w-]+\s+){0,3}$")
NON_ARTIFACT_HEAD = re.compile(
    r"^[\s-]*(?:strateg(?:y|ies)|ambitions?|vision|journey|road\s?maps?|plans?|initiatives?|opportunit\w+|efforts?|"
    r"agenda|aspirations?|priorit\w+|polic(?:y|ies)|principles|frameworks?|governance|readiness|ethics|literacy|"
    r"exposure|risks?|programs?|programmes?|transformation)\b",
    _I,
)
FORWARD = re.compile(
    r"\b(?:will|shall|plan(?:s|ned|ning)?\s+to|intend(?:s|ed)?\s+to|expect(?:s|ed|ing)?|aim(?:s|ed)?\s+to|"
    r"seek(?:s|ing)?\s+to|strive\s+to|anticipate[sd]?|explor(?:e|es|ed|ing|ation)|evaluat(?:e|es|ed|ing)|"
    r"assess(?:es|ed|ing)?\s+(?:the\s+)?(?:use|potential|feasibility|opportunit\w*)|consider(?:s|ed|ing)?|"
    r"pilot(?:s|ed|ing)?|trial(?:s|led|ling)?|experiment(?:s|ed|ing)?|test(?:s|ed|ing)?\s+(?:the\s+)?(?:use|potential)|"
    r"proof[\s-]+of[\s-]+concept|future|going\s+forward|in\s+the\s+coming|road\s?map|ambition|aspir\w+|"
    r"opportunit(?:y|ies)|potential(?:ly)?|beginning\s+to|early\s+stages?|intention|goal\s+(?:is|of)|"
    r"look(?:ing)?\s+(?:forward\s+)?to|hope\s+to|over\s+time)\b",
    _I,
)
STRONG_INTENT = re.compile(r"(?:plan|intend|expect|aim|seek)", _I)
CONTINUE = re.compile(
    r"\b(?:continue[sd]?|continuing)\s+to\s+(?:use|utili|employ|deploy|leverag|invest|expand|scale|develop|integrat|"
    r"enhanc|improv|offer|sell|appl|embed|incorporat|build|train)\w*|\bexpand(?:ed|ing)?\s+(?:our|the)\s+use\b",
    _I,
)
HEDGE = re.compile(
    r"\b(?:may|might|could|would|can|possibly|if|to\s+the\s+extent|no\s+assurance|uncertain(?:ty|ties)?|unknown|whether)\b",
    _I,
)
RISK = re.compile(
    r"\b(?:risks?|adverse(?:ly)?|harm(?:ed|ful|s)?|liabilit(?:y|ies)|litigation|lawsuits?|reputation(?:al)?|errors?|flaws?|"
    r"bias(?:es|ed)?|hallucinat\w*|misuse|breach(?:es)?|vulnerabilit(?:y|ies)|unintended|failures?|fail(?:s|ed)?|"
    r"damage|scrutiny|penalt(?:y|ies)|fines?|challenges|disrupt\w*|threats?|expos(?:e|es|ed|ure)|infring\w*)\b",
    _I,
)
APPLICATION = re.compile(
    r"\b(?:fraud\w*|underwrit\w*|credit\s+(?:scoring|decisions?|decisioning|risk|models?)|risk\s+(?:scoring|assessment|models?)|"
    r"pricing|price|forecast(?:s|ing)?|demand|inventory|supply\s+chain|logistics|routing|routes?|"
    r"recommend(?:ation|ations|er|s)?|personali[sz](?:e|es|ed|ation)|customer\s+(?:service|support|care|interactions|inquiries)|"
    r"call\s+cent(?:er|re)s?|contact\s+cent(?:er|re)s?|claims?\s+(?:processing|handling|triage|adjust\w*)|"
    r"documents?\s+(?:processing|review|extraction)?|(?:code|software)\s+(?:generation|development|completion)|coding|"
    r"drug\s+discovery|compounds?|diagnos(?:is|tic|tics|e)|imaging|radiology|patholog\w*|predictive\s+maintenance|"
    r"quality\s+(?:control|inspection)|defect\s+detection|anomaly\s+detection|cyber\s?security|threat\s+detection|"
    r"search|relevance|advertis\w*|ad\s+targeting|content\s+(?:moderation|generation|creation)|translation|transcription|"
    r"chat\s?bots?|virtual\s+assistants?|(?:AI|digital)\s+assistants?|assistants?|copilots?|agents|autonomous|robot(?:s|ic|ics)?|"
    r"trading|portfolio|marketing|sales|hiring|recruit(?:ing|ment)|scheduling|energy\s+(?:management|optimi\w+)|"
    r"invoices?|invoicing|reconcil\w+|accounts\s+(?:payable|receivable)|procurement|sorting|picking|warehouses?|"
    r"(?:preventive|predictive)?\s?maintenance|"
    r"fuel|yield|crop|manufacturing|production\s+(?:lines?|planning)|seismic|reservoir|grid|"
    r"loans?|lending|mortgages?|payments?|transactions?|anti-?money\s+laundering|AML|compliance\s+monitoring|"
    r"image\s+recognition|speech\s+recognition|voice|vision|inspection|triage|detect(?:s|ed|ing|ion)?|"
    r"classif(?:y|ies|ied|ication)|predict(?:s|ed|ing|ion|ions)?|optimi[sz](?:e|es|ed|ing|ation)|accelerators?|chips?|GPUs?)\b",
    _I,
)
QUANT = re.compile(
    r"(?:\d+(?:\.\d+)?\s?(?:%|percent|per\s+cent)|[$€£¥]\s?\d|\b\d{1,3}(?:,\d{3})+\b|"
    r"\b\d+(?:\.\d+)?\s+(?:million|billion|thousand|hours|days|minutes|employees|customers|users|models|patents|"
    r"countries|languages|sites|plants|stores|branches|engineers|regions|centers|centres)\b|"
    r"\b(?:over|more\s+than|approximately|nearly|about|some)\s+\d+)",
    _I,
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NAMED = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b|\b[A-Z][a-z]{2,}\s+(?:AI|Copilot|Assistant|Assist|GPT)\b")
PASSIVE_DEPLOY = re.compile(
    r"\b(?:is|are|was|were|has\s+been|have\s+been|being)\s+(?:now\s+|also\s+|increasingly\s+|currently\s+)?"
    r"(?:used|deployed|handled|powered|applied|built|developed|implemented|integrated|trained|embedded|automated|run|"
    r"operated|processed|generated|supported|performed)\b|^(?:This|These|It|Such|The\s+(?:system|platform|tool|model|"
    r"engine|solution|technology))\b",
    _I,
)
# Financial-statement subjects make an agentless sentence implicitly about the focal firm
# ("AI-related revenue accounted for 12% of revenue", "capital expenditures for AI infrastructure").
FIN_NOUN = re.compile(
    r"\b(?:capital\s+expenditures?|capex|investments?|spending|expenses?|costs?|revenues?|net\s+sales|bookings|backlog)\b",
    _I,
)
FIN_REVENUE = re.compile(r"^(?:revenues?|net\s+sales|bookings|backlog)$", _I)
_GENERIC_NOUN_USE = re.compile(r"\bthe\s+$", _I)  # "the use of AI" is a noun phrase, not a focal claim
# agentless use/development noun phrase right before the term ("the development and use of AI")
AGENTLESS_USE = re.compile(
    r"\b(?:development|use|usage|deployment|adoption|integration|implementation)"
    r"(?:\s*,?\s*(?:and|or|,)\s*(?:the\s+)?(?:development|use|usage|deployment|adoption|integration|implementation))*"
    r"\s+of\s+(?:[\w-]+\s+){0,2}$",
    _I,
)
# predictions about AI's effect on markets or demand ("how AI will affect demand for our services")
EFFECT_BEFORE = re.compile(
    r"\b(?:how|whether|the\s+extent\s+to\s+which|the\s+(?:impact|effects?|implications|consequences)\s+(?:of|that))"
    r"\s+(?:(?:the|new|emerging|rapid|continued|further)\s+)*(?:[\w-]+\s+){0,1}$",
    _I,
)
_EFFECT_OBJECT = (
    r"(?:the\s+)?(?:(?:overall|future|long-term)\s+)?(?:demand|markets?|industry|industries|competit\w+|pricing|prices|"
    r"economy|labou?r|workforce|employment|jobs|our\s+(?:business|industry|markets?|customers|results|operations|"
    r"revenues?|products|services))"
)
EFFECT_VERB_AFTER = re.compile(
    r"^[^.;]{0,40}?\b(?:affect|impact|change|disrupt|transform|reshape|alter|influence|reduce|erode)\w*\s+"
    + _EFFECT_OBJECT
    + r"|^\s+(?:on|upon)\s+"
    + _EFFECT_OBJECT,
    _I,
)
OUR_RIVALS = re.compile(
    r"\s+(?:(?:main|largest|key|principal|direct)\s+)?(?:competitors?|rivals?|competition|peers)\b", _I
)
USE_IN_FOCAL_AFTER = re.compile(
    r"^\s+(?:[\w-]+\s+){0,2}?(?:in|within|across|throughout|by)\s+(?:our|the\s+(?:company|group)'s)\b"
    r"(?!\s+(?:competitors|customers|clients|vendors|suppliers|partners|industry|industries|markets?)\b)",
    _I,
)
# model names inside broader lexicon terms ("ChatGPT", "GPT-4", "OpenAI") are specific too
NAMED_SURFACE = re.compile(r"^(?:ChatGPT(?:\s+\w+)?|GPT-?\d(?:\.\d)?o?|OpenAI)$")
# "we license AI models from ...": licensing in, which is use, not an offer
LICENSE_IN_AFTER = re.compile(r"^[^.;]{0,80}?\bfrom\b", _I)
PROVIDED_BY_AFTER = re.compile(r"^\s+(?:to\s+us\s+)?by\b", _I)
_COMMON_CAPS = {
    "We",
    "Our",
    "The",
    "Company",
    "Group",
    "In",
    "As",
    "For",
    "This",
    "These",
    "Item",
    "Generative",
    "Conversational",
    "Responsible",
    "Applied",
    "Agentic",
    "Enterprise",
    "Ethical",
    "Trustworthy",
    "Advanced",
    "Artificial",
    "Using",
    "With",
    "New",
    "Such",
    "Many",
    "Through",
    "Its",
    "Machine",
    "Deep",
    "Natural",
}
_GENERIC_FIRST = {
    "first",
    "american",
    "general",
    "united",
    "national",
    "international",
    "global",
    "new",
    "the",
    "great",
    "royal",
    "central",
    "western",
    "eastern",
    "southern",
    "northern",
    "pacific",
    "atlantic",
    "standard",
}


@dataclass
class _HitResult:
    state: AdoptionState
    reason: Reason
    confidence: float
    cues: dict[str, list[str]] = field(default_factory=dict)


def conf_adj_early(inherited: bool) -> float:
    return -0.15 if inherited else 0.0


def _spans(rx: re.Pattern[str], text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in rx.finditer(text)]


def firm_alias_text(firm_name: str) -> str | None:
    """Short form of a firm's name ("Northwind Logistics Inc." -> "Northwind")."""
    if not firm_name:
        return None
    suffix = (
        r"(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|llp|holdings?|group|"
        r"a/s|as|ab|asa|oyj|nv|n\.v|sa|s\.a|se|ag|spa|s\.p\.a|bv|b\.v)\.?"
    )
    name = firm_name.strip()
    for _ in range(3):
        name = re.sub(r"[,.]?\s+" + suffix + r"\s*$", "", name, flags=_I)
    tokens = [t for t in re.split(r"\s+", name) if t]
    if not tokens:
        return None
    if tokens[0].lower() in _GENERIC_FIRST and len(tokens) > 1:
        alias = " ".join(tokens[:2])
    else:
        alias = tokens[0]
    if len(alias) < 3:
        return None
    return alias


def firm_alias_regex(firm_name: str) -> re.Pattern[str] | None:
    """Compile a pattern matching the short form of a firm's name."""
    alias = firm_alias_text(firm_name)
    if alias is None:
        return None
    return re.compile(r"\b" + re.escape(alias) + r"(?:'s)?\b")


def firm_possessive_regex(firm_name: str) -> re.Pattern[str] | None:
    """Pattern for the firm's name in the possessive just before a term ("Delta Bank's ... ")."""
    alias = firm_alias_text(firm_name)
    if alias is None:
        return None
    return re.compile(r"\b" + re.escape(alias) + r"(?:\s+[A-Z][\w&.-]*){0,3}['’]s\s+(?:[\w-]+\s+){0,3}$")


class RuleClassifier(Classifier):
    """Deterministic, auditable baseline classifier.

    Parameters
    ----------
    window:
        Number of characters on each side of an AI term in which deployment,
        specificity and context cues are considered to refer to that term.
    inherit_focal_from_context:
        If a sentence has no agent cue, is written in the passive voice (or
        starts with "This/It/The system ...") and the previous sentence is in
        the focal firm's voice, treat the sentence as focal (lower confidence).
    """

    name = "rules"
    version = "1.1.0"

    def __init__(self, window: int = 110, inherit_focal_from_context: bool = True):
        self.window = window
        self.inherit_focal_from_context = inherit_focal_from_context
        self._alias_cache: dict[str, re.Pattern[str] | None] = {}

    def config(self) -> dict:
        return {
            "backend": self.name,
            "version": self.version,
            "window": self.window,
            "inherit_focal_from_context": self.inherit_focal_from_context,
        }

    # ------------------------------------------------------------------ public
    def classify(self, candidates: Sequence[Candidate]) -> list[Prediction]:
        return [self.classify_one(c) for c in candidates]

    def classify_one(self, cand: Candidate) -> Prediction:
        if not cand.terms:
            return Prediction(
                AdoptionState.NO_CLAIM, Reason.FALSE_POSITIVE, 0.8, self.name, {"filtered_terms": list(cand.filtered)}
            )
        spans = list(cand.term_spans) or _locate_terms(cand)
        results = [self._classify_hit(cand, int(a), int(b)) for a, b in spans]
        best = max(results, key=lambda r: (r.state.rank, r.confidence))
        details = dict(best.cues)
        details["n_hits"] = len(results)
        return Prediction(best.state, best.reason, round(min(max(best.confidence, 0.05), 0.99), 3), self.name, details)

    # ----------------------------------------------------------------- helpers
    def _alias(self, firm_name: str):
        if firm_name not in self._alias_cache:
            self._alias_cache[firm_name] = firm_alias_regex(firm_name)
        return self._alias_cache[firm_name]

    def _alias_possessive(self, firm_name: str):
        key = "\x00possessive:" + firm_name
        if key not in self._alias_cache:
            self._alias_cache[key] = firm_possessive_regex(firm_name)
        return self._alias_cache[key]

    def _classify_hit(self, cand: Candidate, t_start: int, t_end: int) -> _HitResult:
        s = cand.text
        w = self.window
        lo = max(0, t_start - w)
        hi = min(len(s), t_end + w)
        cues: dict[str, list[str]] = {}

        def near(rx: re.Pattern[str], name: str, left: int | None = None, right: int | None = None):
            left = lo if left is None else left
            right = hi if right is None else right
            found = [
                sp
                for sp in _spans(rx, s)
                if sp[1] > left and sp[0] < right and not (sp[0] >= t_start and sp[1] <= t_end)
            ]
            if found:
                cues[name] = [f[2] for f in found]
            return found

        # ---- AI term inside the name of a law or regulator --------------------
        if REG_NAME_AFTER.match(s[t_end : t_end + 40]):
            cues["regulatory_name"] = [s[t_start : min(len(s), t_end + 12)].strip()]
            return _HitResult(AdoptionState.NO_CLAIM, Reason.REGULATORY, 0.8, cues)

        # ---- agent -----------------------------------------------------------
        focal_cues = [
            (m.start(), m.end(), "focal", m.group(0))
            for m in FOCAL.finditer(s)
            # "our competitors": the possessive does not make competitors the focal firm
            if not (m.group(0).lower() == "our" and OUR_RIVALS.match(s, m.end()))
        ]
        alias = self._alias(cand.firm_name)
        if alias is not None:
            focal_cues += [(m.start(), m.end(), "focal", m.group(0)) for m in alias.finditer(s)]
        tp_cues = [(m.start(), m.end(), "third_party", m.group(0)) for m in THIRD_PARTY.finditer(s)]
        agents = [a for a in focal_cues + tp_cues if not (a[0] >= t_start and a[1] <= t_end)]
        before = [a for a in agents if a[1] <= t_start]
        after = [a for a in agents if a[0] >= t_end]
        agent, agent_pos, agent_word, passive = "none", None, "", False
        if before:
            a = max(before, key=lambda x: x[0])
            agent, agent_pos, agent_word = a[2], a[0], a[3]
        elif after:
            a = min(after, key=lambda x: x[0])
            agent, agent_pos, agent_word, passive = a[2], a[0], a[3], True
        if agent_word:
            cues["agent"] = [agent_word]
        service = False
        if agent == "third_party" and not passive and agent_pos is not None:
            if SERVICE_TO_THIRD_PARTY.search(s[:agent_pos]):
                agent, service = "focal", True
                cues["service_to_third_party"] = ["yes"]
            elif ATTRIBUTIVE_TP.search(s[max(0, t_start - 50) : t_start]):
                # "we use third-party AI models ...": the focal firm uses AI supplied by others
                prior_focal = [f for f in focal_cues if f[1] <= agent_pos]
                if prior_focal:
                    f0 = max(prior_focal, key=lambda f: f[0])
                    between = s[f0[1] : agent_pos]
                    if DEPLOY.search(between) or OFFER.search(between):
                        agent, agent_pos, agent_word = "focal", f0[0], f0[3]
                        cues["agent"] = [agent_word]
                        cues["third_party_supplied"] = ["yes"]
        inherited = False
        financial: list[tuple[int, int, str]] = []
        if agent == "none":
            financial = [sp for sp in _spans(FIN_NOUN, s) if abs(sp[0] - t_start) <= 40 or abs(sp[1] - t_end) <= 40]
            if financial and not MARKET.search(s) and not REGULATORY.search(s):
                agent, inherited = "focal", True
                cues["agent"] = ["<financial statement subject>"]
                cues["financial"] = [f[2] for f in financial]
            else:
                financial = []
        if agent == "none" and self.inherit_focal_from_context and cand.prev_text:
            if FOCAL.search(cand.prev_text) and PASSIVE_DEPLOY.search(s) and not MARKET.search(s):
                agent, inherited = "focal", True
                cues["agent"] = ["<previous sentence>"]

        # ---- cues ------------------------------------------------------------
        regulatory = near(REGULATORY, "regulatory")
        market = near(MARKET, "market")
        risk = [sp[2] for sp in _spans(RISK, s)]
        if risk:
            cues["risk"] = risk
        risk_context = is_risk_section(cand.section) or len(risk) >= 2 or (len(risk) >= 1 and bool(HEDGE.search(s)))

        if agent == "third_party":
            reason = Reason.REGULATORY if REGULATOR_AGENT.match(agent_word.strip()) else Reason.THIRD_PARTY
            return _HitResult(AdoptionState.NO_CLAIM, reason, 0.75, cues)
        if agent == "none":
            # Risk-factor boilerplate about AI in general ("Issues in the development
            # and use of AI ... may result in reputational harm") reads as the
            # firm's own risk disclosure (codebook v1.1, rule 10).
            agentless_use = AGENTLESS_USE.search(s[max(0, t_start - 80) : t_start])
            if (
                agentless_use
                and is_risk_section(cand.section)
                and risk
                and HEDGE.search(s)
                and not market
                and not REG_NAME_AFTER.match(s[t_end : t_end + 40])
                and not re.search(r"subject\s+to\s+(?:\w+\s+){0,3}(?:regulat|scrutiny|laws?)", s, _I)
            ):
                cues["agentless_use"] = [agentless_use.group(0).strip()]
                return _HitResult(AdoptionState.WEAK, Reason.RISK_BOILERPLATE, 0.5, cues)
            if regulatory:
                return _HitResult(AdoptionState.NO_CLAIM, Reason.REGULATORY, 0.7, cues)
            if market:
                return _HitResult(AdoptionState.NO_CLAIM, Reason.MARKET_TREND, 0.7, cues)
            return _HitResult(AdoptionState.NO_CLAIM, Reason.GENERIC, 0.55, cues)

        # The focal agent's clause runs from the agent to the term (active) or
        # from the term to the agent (passive).
        if inherited or agent_pos is None:
            clause = (lo, t_start)
        elif passive:
            clause = (t_end, agent_pos)
        else:
            clause = (agent_pos, t_start)
        negation = near(NEGATION, "negation", clause[0], max(clause[1], t_start) + (0 if passive else 25))
        if negation:
            return _HitResult(AdoptionState.NO_CLAIM, Reason.NEGATED, 0.75 - (0.15 if inherited else 0), cues)

        deploy = near(DEPLOY, "deployment")
        offer = near(OFFER, "offer")
        # licensing in ("we license models from X") and "provided by" are use, not offers
        licensed_in = [
            o
            for o in offer
            if (o[2].lower().startswith("licens") and LICENSE_IN_AFTER.match(s[o[1] :]))
            or (o[2].lower().startswith("provid") and PROVIDED_BY_AFTER.match(s[o[1] :]))
        ]
        if licensed_in:
            offer = [o for o in offer if o not in licensed_in]
            deploy = sorted(deploy + licensed_in)
            cues["licensed_in"] = [o[2] for o in licensed_in]
            if offer:
                cues["offer"] = [o[2] for o in offer]
            else:
                cues.pop("offer", None)
        invest = near(INVEST, "investment")
        cont = near(CONTINUE, "continue")
        product_adj = near(PRODUCT_ADJ, "product_adj")
        if not product_adj and PRODUCT_ADJ.match(s, t_start):
            product_adj = [(t_start, t_end, s[t_start:t_end])]
            cues["product_adj"] = [s[t_start : min(len(s), t_end + 12)]]
        use_noun = near(USE_NOUN, "use_noun", lo, t_start)
        if not use_noun and AGENTLESS_USE.search(s[max(0, t_start - 80) : t_start]):
            m_in = USE_IN_FOCAL_AFTER.match(s[t_end:])
            if m_in:  # "the use of AI in our underwriting"
                use_noun = [(t_end + m_in.start(), t_end + m_in.end(), m_in.group(0).strip())]
                cues["use_noun"] = [m_in.group(0).strip()]
        pre = s[max(0, t_start - 80) : t_start]
        alias_pos = self._alias_possessive(cand.firm_name)
        possessive = bool(POSSESSIVE_BEFORE.search(pre) or (alias_pos is not None and alias_pos.search(pre))) and not (
            NON_ARTIFACT_HEAD.match(s[t_end : t_end + 40])
        )
        if possessive:
            cues["possessive"] = ["yes"]

        # action verbs linked to the focal firm and the term
        focal_before = sorted(a[0] for a in focal_cues if a[1] <= t_start)
        tp_before = sorted(a[0] for a in tp_cues if a[1] <= t_start)
        if "third_party_supplied" in cues:  # the attributive "third-party" does not block the link
            tp_before = [tp for tp in tp_before if tp < (agent_pos or 0)]
        candidates_actions = [
            a
            for a in deploy + offer + invest
            if not _GENERIC_NOUN_USE.search(s[max(0, a[0] - 5) : a[0]])
            and not any(r[0] <= a[0] < r[1] for r in regulatory)
            and not (regulatory and re.match(r"appl(?:y|ies|ied|ying)$", a[2], _I))  # "the law applies to ..."
        ]
        action = []
        for a in candidates_actions:
            if passive:
                ok = t_end <= a[0] <= agent_pos
            elif a[0] < t_start:
                ok = (
                    (t_start - a[0]) <= w
                    and (inherited or any(f <= a[0] for f in focal_before))
                    and not any(a[0] < tp < t_start for tp in tp_before)
                )
            else:  # verb after the term: only when the term sits in a focal noun phrase
                ok = (possessive or bool(product_adj) or inherited) and (a[0] - t_end) <= w
            if ok:
                action.append(a)
        action.sort(key=lambda a: a[0])
        if action:
            cues["linked_action"] = [a[2] for a in action]
        claims_use = bool(action or use_noun or product_adj or possessive or service or financial)

        if not claims_use and market:
            return _HitResult(AdoptionState.NO_CLAIM, Reason.MARKET_TREND, 0.6, cues)
        # predictions about AI's effect on markets, demand or the firm (codebook rule 8)
        effect_before = EFFECT_BEFORE.search(s[max(0, t_start - 60) : t_start])
        if (
            effect_before
            and EFFECT_VERB_AFTER.match(s[t_end:])
            and not (possessive or product_adj or use_noun or financial)
            and not any(a[0] < t_start for a in action)
        ):
            cues["effect_prediction"] = [effect_before.group(0).strip()]
            return _HitResult(AdoptionState.NO_CLAIM, Reason.MARKET_TREND, 0.6 + conf_adj_early(inherited), cues)
        if not claims_use and regulatory:
            return _HitResult(AdoptionState.NO_CLAIM, Reason.REGULATORY, 0.65, cues)

        # forward-looking and hedging cues between the focal agent and the claim
        first_action = action[0][0] if action else t_start
        if inherited or passive or agent_pos is None:
            anchor = lo
        else:
            anchor = max([f for f in focal_before if f <= first_action] or [agent_pos])
            anchor = min(anchor, agent_pos)
        forward_left = [sp for sp in _spans(FORWARD, s) if anchor <= sp[0] < max(first_action, t_start)]
        forward_right = [sp for sp in _spans(FORWARD, s) if t_end <= sp[0] < hi and STRONG_INTENT.match(sp[2])]
        if forward_left or forward_right:
            cues["forward"] = [f[2] for f in forward_left + forward_right]
        hedge_left = [sp for sp in _spans(HEDGE, s) if anchor <= sp[0] < max(first_action, t_start)]
        if hedge_left:
            cues["hedge"] = [h[2] for h in hedge_left]
        forward_governs = bool(forward_left or forward_right) and not cont
        hedge_right = []
        if (use_noun or possessive or product_adj or financial) and not any(a[0] < t_start for a in action):
            m_after = MODAL_AFTER.search(s[t_end:])
            if m_after:
                hedge_right = [(t_end + m_after.start(), t_end + m_after.end(), m_after.group(0).split()[-1])]
                cues.setdefault("hedge", []).append(hedge_right[0][2])
        hedge_governs = bool(hedge_left or hedge_right)

        application = near(APPLICATION, "application")
        quant = near(QUANT, "quantity")
        year = near(YEAR, "date")
        named = [sp for sp in _spans(NAMED, s) if sp[1] > lo and sp[0] < hi and sp[2].split()[0] not in _COMMON_CAPS]
        named = [sp for sp in named if not (sp[0] >= t_start and sp[1] <= t_end)]
        named_rx = _named_model_regex()
        surface = s[t_start:t_end]
        if (named_rx is not None and named_rx.fullmatch(surface)) or NAMED_SURFACE.match(surface):
            named.append((t_start, t_end, surface))  # a named AI model is itself specific
        if named:
            cues["named_product"] = [n[2] for n in named]
        specific = bool(application or quant or year or named or invest or (financial and (quant or year)))
        conf_adj = -0.15 if inherited else 0.0

        # ---- decision procedure ---------------------------------------------
        if regulatory and not (action and specific):
            if claims_use:
                reason = Reason.RISK_BOILERPLATE if risk_context else Reason.VAGUE
                return _HitResult(AdoptionState.WEAK, reason, 0.6 + conf_adj, cues)
            return _HitResult(AdoptionState.NO_CLAIM, Reason.REGULATORY, 0.65 + conf_adj, cues)
        if forward_governs:
            return _HitResult(AdoptionState.WEAK, Reason.FORWARD_LOOKING, 0.75 + conf_adj, cues)
        if hedge_governs:
            reason = Reason.RISK_BOILERPLATE if risk_context else Reason.HEDGED
            return _HitResult(AdoptionState.WEAK, reason, 0.65 + conf_adj, cues)
        if claims_use and specific:
            external = bool(EXTERNAL_USER.search(s)) and bool(PRODUCT_NOUN.search(s))
            if financial:
                reason = Reason.PRODUCT if any(FIN_REVENUE.match(f[2]) for f in financial) else Reason.INVESTMENT
            elif any(INVEST_ACT.match(i[2]) for i in invest) and not offer:
                reason = Reason.INVESTMENT
            elif offer or service or external:
                reason = Reason.PRODUCT
            else:
                reason = Reason.DEPLOYMENT
            n_specific = sum(bool(x) for x in (application, quant, year, named, invest))
            conf = (0.7 if n_specific == 1 else 0.85) + conf_adj
            return _HitResult(AdoptionState.SUBSTANTIVE, reason, conf, cues)
        if risk_context:
            return _HitResult(AdoptionState.WEAK, Reason.RISK_BOILERPLATE, 0.6 + conf_adj, cues)
        return _HitResult(AdoptionState.WEAK, Reason.VAGUE, 0.55 + conf_adj, cues)


def _named_model_regex() -> re.Pattern[str] | None:
    """The lexicon's ``named_ai_model`` pattern: a named model is a specific element."""
    if "named" not in _LEX_CACHE:
        lex = _default_lexicon()
        term = next((t for t in lex.terms if t.id == "named_ai_model"), None)
        _LEX_CACHE["named"] = term.compile() if term is not None else None
    return _LEX_CACHE["named"]  # type: ignore[return-value]


def _locate_terms(cand: Candidate) -> list[tuple[int, int]]:
    """Find term offsets when a candidate was built without them."""
    lex = _default_lexicon()
    hits = lex.find(cand.text)
    wanted = set(cand.terms)
    pos = [(h.start, h.end) for h in hits.valid if h.term_id in wanted]
    return pos or [(0, len(cand.text))]


_LEX_CACHE: dict[str, object] = {}


def _default_lexicon():
    from ..lexicon import Lexicon

    if "default" not in _LEX_CACHE:
        _LEX_CACHE["default"] = Lexicon.load("default", tiers=None)
    return _LEX_CACHE["default"]
