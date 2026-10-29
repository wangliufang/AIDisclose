"""Section segmentation for annual reports.

Item-structured SEC forms (10-K, 20-F)
--------------------------------------
Item headings usually appear twice, once in the table of contents and once at
the start of each section, and cross-references ("Item 7 of this report") can
look like headings. :func:`split_items` therefore

1. finds all heading lines, including combined headings ("Items 1 and 2.") and
   run-in headings ("Item 1. Business. The Company ...");
2. takes the body to start at the first-Item heading that opens the longest
   span (the table-of-contents entry opens a short span); and
3. after that point keeps the longest sequence of headings whose Item numbers
   increase, which discards cross-references and stray matches.

Free-form annual reports (ESEF, UKSEF, PDFs)
-------------------------------------------
:func:`split_annual_report` recognises the headings of the usual parts of a
European or international annual report -- management report, risk report,
sustainability statement, corporate governance and remuneration, financial
statements and notes, auditor's report -- and assigns each block of text to
the most recent part. A heading counts only if at least ``min_words`` words
follow it before the next heading, which ignores tables of contents and
running headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------ schemas
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved / Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
_TITLE_START_10K = {
    "1": r"business",
    "1A": r"risk\s+factors",
    "1B": r"unresolved",
    "1C": r"cybersecurity",
    "2": r"properties",
    "3": r"legal\s+proceedings",
    "4": r"mine\s+safety",
    "5": r"market\s+for",
    "6": r"(?:\[?reserved|selected)",
    "7": r"management",
    "7A": r"quantitative",
    "8": r"financial\s+statements",
    "9": r"changes\s+in",
    "9A": r"controls",
    "9B": r"other\s+information",
    "9C": r"disclosure\s+regarding",
    "10": r"directors",
    "11": r"executive\s+compensation",
    "12": r"security\s+ownership",
    "13": r"certain\s+relationships",
    "14": r"principal\s+account",
    "15": r"exhibits",
    "16": r"form\s+10-k\s+summary",
}

ITEM_TITLES_20F = {
    "1": "Identity of Directors, Senior Management and Advisers",
    "2": "Offer Statistics and Expected Timetable",
    "3": "Key Information (incl. 3.D Risk Factors)",
    "4": "Information on the Company",
    "4A": "Unresolved Staff Comments",
    "5": "Operating and Financial Review and Prospects",
    "6": "Directors, Senior Management and Employees",
    "7": "Major Shareholders and Related Party Transactions",
    "8": "Financial Information",
    "9": "The Offer and Listing",
    "10": "Additional Information",
    "11": "Quantitative and Qualitative Disclosures About Market Risk",
    "12": "Description of Securities Other than Equity Securities",
    "13": "Defaults, Dividend Arrearages and Delinquencies",
    "14": "Material Modifications to the Rights of Security Holders",
    "15": "Controls and Procedures",
    "16": "[Reserved]",
    "16A": "Audit Committee Financial Expert",
    "16B": "Code of Ethics",
    "16C": "Principal Accountant Fees and Services",
    "16D": "Exemptions from the Listing Standards for Audit Committees",
    "16E": "Purchases of Equity Securities by the Issuer",
    "16F": "Change in Registrant's Certifying Accountant",
    "16G": "Corporate Governance",
    "16H": "Mine Safety Disclosure",
    "16I": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "16J": "Insider Trading Policies",
    "16K": "Cybersecurity",
    "17": "Financial Statements",
    "18": "Financial Statements",
    "19": "Exhibits",
}
_TITLE_START_20F = {
    "1": r"identity",
    "2": r"offer\s+statistics",
    "3": r"key\s+information",
    "4": r"information\s+on",
    "4A": r"unresolved",
    "5": r"operating\s+and\s+financial",
    "6": r"directors",
    "7": r"major\s+shareholders",
    "8": r"financial\s+information",
    "9": r"the\s+offer",
    "10": r"additional\s+information",
    "11": r"quantitative",
    "12": r"description\s+of\s+securities",
    "13": r"defaults",
    "14": r"material\s+modifications",
    "15": r"controls",
    "16": r"\[?reserved",
    "16A": r"audit\s+committee",
    "16B": r"code\s+of\s+ethics",
    "16C": r"principal\s+account",
    "16D": r"exemptions",
    "16E": r"purchases",
    "16F": r"change\s+in",
    "16G": r"corporate\s+governance",
    "16H": r"mine\s+safety",
    "16I": r"disclosure\s+regarding",
    "16J": r"insider\s+trading",
    "16K": r"cybersecurity",
    "17": r"financial\s+statements",
    "18": r"financial\s+statements",
    "19": r"exhibits",
}


@dataclass(frozen=True)
class ItemSchema:
    form: str
    titles: dict[str, str]
    title_start: dict[str, str]
    first_item: str = "1"
    risk_items: tuple[str, ...] = ()

    @property
    def order(self) -> dict[str, int]:
        return {code: i for i, code in enumerate(self.titles)}


SCHEMAS = {
    "10-K": ItemSchema("10-K", ITEM_TITLES, _TITLE_START_10K, risk_items=("1A",)),
    "20-F": ItemSchema("20-F", ITEM_TITLES_20F, _TITLE_START_20F, risk_items=("3",)),
}
ITEM_ORDER = SCHEMAS["10-K"].order

_HEADING = re.compile(
    r"^(?:part\s+[ivx]+\s*[,.:\-–—]?\s*)?items?\s*(\d{1,2}[a-k]?)"
    r"(?:\s*(?:,|and|&)\s*\d{1,2}[a-k]?)*\s*(?:[.:\-–—]|\s|$)",
    re.IGNORECASE,
)
_MAX_HEADING_LEN = 160


def schema_for(form: str | None) -> ItemSchema | None:
    """Return the Item schema of an SEC form ("10-K", "10-K405", "20-F", ...)."""
    if not form:
        return None
    f = form.upper().replace("/A", "")
    if f.startswith("10-K"):
        return SCHEMAS["10-K"]
    if f.startswith("20-F"):
        return SCHEMAS["20-F"]
    return None


def find_item_headings(text: str, schema: ItemSchema | None = None) -> list[tuple[int, str]]:
    """Return ``(offset, item_code)`` for every line that looks like an Item heading."""
    schema = schema or SCHEMAS["10-K"]
    out = []
    pos = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            m = _HEADING.match(stripped)
            if m:
                code = m.group(1).upper()
                if code in schema.titles:
                    ok = len(stripped) <= _MAX_HEADING_LEN
                    if not ok:  # long line: accept only a run-in heading with the canonical title
                        rest = stripped[m.end() :].lstrip(" .:-–—")
                        ok = bool(re.match(schema.title_start[code], rest, re.IGNORECASE))
                    if ok:
                        out.append((pos + (len(line) - len(line.lstrip())), code))
        pos += len(line) + 1
    return out


def _longest_increasing(headings: list[tuple[int, str]], order: dict[str, int]) -> list[tuple[int, str]]:
    """Longest subsequence with strictly increasing Item order (ties: latest end)."""
    n = len(headings)
    if n == 0:
        return []
    ranks = [order[c] for _, c in headings]
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if ranks[j] < ranks[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    k = max(range(n), key=lambda i: (best[i], i))
    chain = []
    while k != -1:
        chain.append(headings[k])
        k = prev[k]
    return chain[::-1]


def split_items(text: str, form: str = "10-K", min_items: int = 3) -> dict[str, str]:
    """Split an Item-structured SEC form into ``{"<form>:<item>": text}``.

    Returns ``{"full": text}`` when fewer than ``min_items`` distinct headings
    are found. Text before the first selected heading (cover page, table of
    contents, forward-looking statements) is returned as ``"<form>:cover"``.
    Combined headings such as "Items 1 and 2" are labelled with the first Item.
    """
    schema = schema_for(form) or SCHEMAS["10-K"]
    headings = find_item_headings(text, schema)
    if len({code for _, code in headings}) < min_items:
        return {"full": text}
    body_start = 0
    first = [i for i, (_, c) in enumerate(headings) if c == schema.first_item]
    if first:

        def span(i: int) -> int:
            end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            return end - headings[i][0]

        body_start = headings[max(first, key=span)][0]
    body = [h for h in headings if h[0] >= body_start]
    chain = _longest_increasing(body, schema.order)
    if len(chain) < min_items:
        return {"full": text}
    sections: dict[str, str] = {}
    cover = text[: chain[0][0]].strip()
    if cover:
        sections[f"{schema.form}:cover"] = cover
    for i, (start, code) in enumerate(chain):
        end = chain[i + 1][0] if i + 1 < len(chain) else len(text)
        part = text[start:end].strip()
        if part:
            sections[f"{schema.form}:{code}"] = part
    return sections


def split_10k_items(text: str, min_items: int = 3) -> dict[str, str]:
    """Split a Form 10-K into ``{"10-K:<item>": text}`` (see :func:`split_items`)."""
    return split_items(text, "10-K", min_items=min_items)


def split_20f_items(text: str, min_items: int = 3) -> dict[str, str]:
    """Split a Form 20-F into ``{"20-F:<item>": text}`` (see :func:`split_items`)."""
    return split_items(text, "20-F", min_items=min_items)


# --------------------------------------------------------- annual reports
AR_PARTS: dict[str, re.Pattern[str]] = {
    "auditor": re.compile(
        r"^(?:independent\s+auditor'?s'?\s+report|report\s+of\s+(?:the\s+)?independent\s+(?:registered\s+)?"
        r"(?:public\s+accounting\s+firm|auditors?)|auditor'?s'?\s+report|statutory\s+auditor'?s'?\s+report)",
        re.I,
    ),
    "financial_statements": re.compile(
        r"^(?:(?:consolidated|group|parent\s+company)\s+(?:financial\s+statements|income\s+statements?|statements?\s+of|"
        r"balance\s+sheets?)|notes\s+to\s+the\s+(?:consolidated\s+|group\s+|parent\s+company\s+)?financial\s+statements|"
        r"financial\s+statements$|primary\s+statements$)",
        re.I,
    ),
    "sustainability": re.compile(
        r"^(?:sustainability\s+(?:statement|report|review|information)|non-financial\s+(?:statement|information|report)|"
        r"esg\s+(?:report|review|statement)|corporate\s+(?:social\s+)?responsibility|(?:esrs|csrd)\b|"
        r"environmental,\s+social\s+and\s+governance|statutory\s+(?:report\s+on\s+)?(?:corporate\s+)?social)",
        re.I,
    ),
    "governance": re.compile(
        r"^(?:corporate\s+governance|governance\s+(?:report|statement|review)|(?:directors'?\s+)?remuneration\s+"
        r"(?:report|policy|statement)|statement\s+by\s+(?:the\s+)?(?:board|management|executive)|"
        r"management'?s\s+statement|board\s+of\s+directors\s+and\s+executive)",
        re.I,
    ),
    "risk": re.compile(
        r"^(?:principal\s+risks|risk\s+management|risk\s+factors|risks\s+and\s+uncertainties|risk\s+review|"
        r"(?:key|significant|material)\s+risks|internal\s+control\s+and\s+risk)",
        re.I,
    ),
    "management": re.compile(
        r"^(?:management(?:'s)?\s+(?:review|report|commentary|discussion)|directors'?\s+report|strategic\s+report|"
        r"(?:report|review)\s+of\s+the\s+board|board\s+of\s+directors'?\s+report|letter\s+(?:from|to)\s+(?:the|our)\s+"
        r"(?:ceo|chief\s+executive|chair|shareholders)|(?:ceo|chief\s+executive|chair(?:man|person|woman)?)'?s?\s+"
        r"(?:letter|statement|review|introduction)|our\s+(?:business|strategy)|business\s+(?:review|model|overview)|"
        r"operating\s+and\s+financial\s+review|financial\s+review|(?:the\s+)?year\s+in\s+review|highlights)",
        re.I,
    ),
}
_AR_NUMBERING = re.compile(r"^(?:\d{1,3}(?:\.\d{1,2})*[.)]?|[ivx]{1,5}[.)]|[a-h][.)])\s+", re.I)
_AR_MAX_HEADING = 100


def _ar_part(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > _AR_MAX_HEADING or s.endswith((".", ";", ",")):
        return None
    s = _AR_NUMBERING.sub("", s)
    for part, rx in AR_PARTS.items():
        if rx.match(s):
            return part
    return None


def split_annual_report(text: str, min_words: int = 30, min_parts: int = 2) -> dict[str, str]:
    """Split a free-form annual report into ``{"AR:<part>": text}``.

    Parts: ``front`` (text before the first recognised heading),
    ``management``, ``risk``, ``sustainability``, ``governance``,
    ``financial_statements`` and ``auditor``. Non-contiguous blocks of the same
    part are concatenated in document order. Returns ``{"full": text}`` when
    fewer than ``min_parts`` headings are recognised.
    """
    lines = text.split("\n")
    candidates: list[tuple[int, str]] = [(i, p) for i, line in enumerate(lines) if (p := _ar_part(line))]
    accepted: list[tuple[int, str]] = []
    for k, (i, part) in enumerate(candidates):
        nxt = candidates[k + 1][0] if k + 1 < len(candidates) else len(lines)
        words = sum(len(lines[j].split()) for j in range(i + 1, nxt))
        if words >= min_words:
            accepted.append((i, part))
    if len(accepted) < min_parts:
        return {"full": text}
    blocks: dict[str, list[str]] = {}
    front = "\n".join(lines[: accepted[0][0]]).strip()
    if front:
        blocks.setdefault("AR:front", []).append(front)
    for k, (i, part) in enumerate(accepted):
        end = accepted[k + 1][0] if k + 1 < len(accepted) else len(lines)
        body = "\n".join(lines[i:end]).strip()
        if body:
            blocks.setdefault(f"AR:{part}", []).append(body)
    return {key: "\n\n".join(parts) for key, parts in blocks.items()}


def split_sections(text: str, form: str | None = None) -> dict[str, str]:
    """Split by Item schema for SEC forms, otherwise by annual-report parts."""
    if schema_for(form) is not None:
        return split_items(text, form or "10-K")
    return split_annual_report(text)


# --------------------------------------------------------------- selection
SECTION_PRESETS: dict[str, tuple[str, ...]] = {
    # Business description, risk factors and MD&A (and their 20-F, 40-F and
    # annual-report counterparts): where firms describe what they do.
    "core": (
        "10-K:1",
        "10-K:1A",
        "10-K:7",
        "20-F:3",
        "20-F:4",
        "20-F:5",
        "40-F:AIF",
        "40-F:MDA",
        "AR:front",
        "AR:management",
        "AR:risk",
    ),
    # All narrative sections: core plus sustainability, governance and the
    # 10-K/20-F cybersecurity Items; excludes financial statements, auditor's
    # reports, exhibit indexes and cover pages.
    "narrative": (
        "10-K:1",
        "10-K:1A",
        "10-K:1C",
        "10-K:7",
        "10-K:7A",
        "10-K:9A",
        "10-K:10",
        "20-F:3",
        "20-F:4",
        "20-F:5",
        "20-F:6",
        "20-F:11",
        "20-F:15",
        "20-F:16G",
        "20-F:16K",
        "40-F:AIF",
        "40-F:MDA",
        "AR:front",
        "AR:management",
        "AR:risk",
        "AR:sustainability",
        "AR:governance",
    ),
}
_DUP_SUFFIX = re.compile(r"\(\d+\)$")


class SectionFilter:
    """Decide which section labels a pipeline processes.

    A filter restricts only the document types it mentions: with
    ``["10-K:1A"]`` a 10-K is limited to Item 1A while 20-F filings and
    annual reports are processed in full. Unsplit documents (``"full"``) and
    the main text of filings whose Items could not be found (``"<form>:main"``)
    are always processed. Exhibit 13 (``"10-K:EX-13"``), which carries MD&A
    incorporated by reference, is selected whenever ``"10-K:7"`` is.
    ``None`` or an empty list selects everything. Preset names from
    :data:`SECTION_PRESETS` (``"core"``, ``"narrative"``) expand to their
    labels and can be mixed with labels, e.g. ``["core", "AR:sustainability"]``.
    """

    def __init__(self, sections=None):
        if isinstance(sections, str):
            sections = [s.strip() for s in sections.split(",") if s.strip()]
        labels: list[str] = []
        for item in sections or []:
            for label in SECTION_PRESETS.get(item, (item,)):
                if label not in labels:
                    labels.append(label)
        self.labels = labels
        selected = set(self.labels)
        if "10-K:7" in selected:
            selected.add("10-K:EX-13")
        self._selected = selected
        self._prefixes = {label.partition(":")[0] for label in selected if ":" in label}

    def __bool__(self) -> bool:
        return bool(self.labels)

    def __call__(self, label: str) -> bool:
        if not self.labels:
            return True
        prefix, sep, code = label.partition(":")
        if not sep or prefix not in self._prefixes or code == "main":
            return True
        return _DUP_SUFFIX.sub("", label) in self._selected


# ------------------------------------------------------------------ labels
_EXTRA_TITLES = {
    "main": "Main document",
    "cover": "Cover and table of contents",
    "EX-13": "Exhibit 13 (annual report)",
    "AIF": "Annual Information Form",
    "MDA": "Management's Discussion and Analysis",
}


def section_title(label: str) -> str:
    form, _, code = label.partition(":")
    code = _DUP_SUFFIX.sub("", code)
    if code in _EXTRA_TITLES:
        return _EXTRA_TITLES[code]
    schema = SCHEMAS.get(form)
    if schema is not None:
        return schema.titles.get(code, code)
    if form == "AR":
        return code.replace("_", " ").capitalize()
    return label


def is_risk_section(label: str) -> bool:
    form, _, code = label.partition(":")
    schema = SCHEMAS.get(form)
    if schema is not None and code in schema.risk_items:
        return True
    return "risk" in label.lower()
