"""Plain-text extraction from HTML, inline XBRL (XHTML), PDF and text filings.

Extraction has three steps:

1. **Parsing** (lxml): scripts, styles, hidden inline-XBRL headers
   (``ix:header``), ``display:none`` elements, embedded ``data:`` images and
   mostly numeric tables are removed; block elements become line breaks.
2. **Page-artifact removal**: page numbers, "Table of Contents" links and
   running headers or footers that repeat throughout the document.
3. **Reflow**: many reports place every printed line in its own block (ESEF
   reports converted from PDF, PDF text, 10-K page breaks). Lines are joined
   into paragraphs when the first does not end a sentence and the second
   continues it (lower-case start, or a connector word such as "of" or "and"
   at the end of the first), so that sentences are not cut into fragments.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "tr",
    "li",
    "ul",
    "ol",
    "table",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "footer",
    "blockquote",
    "pre",
    "dd",
    "dt",
    "dl",
    "hr",
    "title",
    "caption",
    "page",
    "body",
    "center",
    "form",
    "fieldset",
    "figure",
    "figcaption",
    "main",
    "nav",
    "aside",
    "address",
}
CELL_TAGS = {"td", "th"}
_REMOVE_TAGS = {"script", "style", "noscript", "head", "ix:header", "xbrli:context", "svg", "object", "embed"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_NUMERIC_CELL = re.compile(r"^[\s$€£¥%()\-–—.,:;*0-9]*$")
_AI_TABLE_TERM = re.compile(
    r"\b(?:artificial intelligence|machine learning|generative ai|gen(?:erative)?\s+ai|"
    r"large language models?|neural networks?|computer vision|predictive analytics|"
    r"natural language processing|\b(?:ai|ml)\b)\b",
    re.I,
)
_WS = re.compile(r"[ \t\r\f\v  -​  　]+")
_DATA_URI = re.compile(r"""(\s(?:src|href|xlink:href)\s*=\s*)(["'])data:[^"']*\2""", re.I)
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)

# ------------------------------------------------------------------ artifacts
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?[-–—]?\s*(?:\d{1,4}|[ivxlc]{1,7})\s*[-–—]?(?:\s*(?:of|/)\s*\d{1,4})?$", re.I)
_TOC_LINK = re.compile(
    r"^(?:back\s+to\s+|return\s+to\s+|go\s+to\s+)?(?:the\s+)?(?:table\s+of\s+)?contents\.?$|^index\s+to\s+financial\s+statements$",
    re.I,
)
_PROTECTED_LINE = re.compile(
    r"^(?:part\s+[ivx]+|items?\s*\d|none\.?$|not\s+applicable)|risk|notes?\s+to\s+the|auditor|financial\s+statements|"
    r"management|directors|governance|remuneration|sustainability|strategic\s+report|annual\s+information\s+form",
    re.I,
)

# --------------------------------------------------------------------- reflow
_TERMINAL = re.compile(r"""[.!?:;]["'”’)\]]*$""")
_ITEM_LINE = re.compile(r"^(?:part\s+[ivx]+\b|items?\s*\d{1,2}[abc]?\b)", re.I)
_BULLET = re.compile(r"^(?:[•▪●○◦■□►✓\-–—*]\s|\(?[a-z0-9]{1,3}[.)]\s)")
_CONNECTORS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "and",
    "or",
    "but",
    "nor",
    "that",
    "which",
    "who",
    "whose",
    "whom",
    "our",
    "its",
    "their",
    "his",
    "her",
    "as",
    "than",
    "into",
    "onto",
    "over",
    "under",
    "across",
    "between",
    "among",
    "per",
    "via",
    "through",
    "within",
    "without",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "will",
    "would",
    "may",
    "might",
    "can",
    "could",
    "should",
    "shall",
    "must",
    "not",
    "also",
    "such",
    "these",
    "those",
    "this",
    "all",
    "any",
    "each",
    "both",
    "more",
    "most",
    "other",
    "including",
    "approximately",
    "about",
    "while",
    "when",
    "where",
    "if",
    "because",
    "since",
    "after",
    "before",
    "during",
    "we",
    "us",
    "it",
    "they",
    "using",
    "use",
    "uses",
    "used",
}


def _is_numeric_table(cells: list[str], threshold: float = 0.5) -> bool:
    cells = [c for c in cells if c]
    if len(cells) < 4:
        return False
    numeric = sum(1 for c in cells if _NUMERIC_CELL.match(c) and any(ch.isdigit() for ch in c))
    return numeric / len(cells) >= threshold


def _tag(el) -> str:
    t = el.tag
    return t.lower() if isinstance(t, str) else ""


def _lxml_text(html: str, drop_numeric_tables: bool, drop_hidden: bool) -> str:
    import lxml.etree as etree
    import lxml.html

    root = lxml.html.document_fromstring(html)
    to_drop = []
    for el in root.iter():
        tag = _tag(el)
        if not tag:
            continue
        if tag in _REMOVE_TAGS or tag.endswith(":header"):
            to_drop.append(el)
        elif drop_hidden and _HIDDEN_STYLE.search(el.get("style") or ""):
            to_drop.append(el)
        elif tag == "img":
            to_drop.append(el)
    for el in to_drop:
        if el.getparent() is not None:
            el.drop_tree()  # keeps the element's tail text
    if drop_numeric_tables:
        for table in list(root.iter("table")):
            if table.getparent() is None:
                continue
            cells = [" ".join(c.text_content().split()) for c in table.iter("td", "th")]
            if _is_numeric_table(cells) and not _AI_TABLE_TERM.search(" ".join(cells)):
                table.drop_tree()
    out: list[str] = []
    for event, el in etree.iterwalk(root, events=("start", "end")):
        tag = _tag(el)
        if event == "start":
            if tag in BLOCK_TAGS:
                out.append("\n")
            if tag and el.text:
                out.append(el.text)
        else:
            if tag in CELL_TAGS:
                out.append(" ")
            if tag in BLOCK_TAGS:
                out.append("\n")
            if el.tail:
                out.append(el.tail)
    return "".join(out)


def _bs4_text(html: str, drop_numeric_tables: bool, drop_hidden: bool) -> str:  # pragma: no cover - fallback
    from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "html.parser")
    for name in _REMOVE_TAGS | {"img"}:
        for tag in soup.find_all(name):
            tag.decompose()
    if drop_hidden:
        for tag in soup.find_all(style=_HIDDEN_STYLE):
            tag.decompose()
    if drop_numeric_tables:
        for table in soup.find_all("table"):
            if table.parent is not None:
                cells = [c.get_text(" ", strip=True) for c in table.find_all(["td", "th"])]
                if _is_numeric_table(cells) and not _AI_TABLE_TERM.search(" ".join(cells)):
                    table.decompose()
    for tag in soup.find_all(["td", "th"]):
        tag.append(NavigableString(" "))
    for tag in soup.find_all(True):
        if tag.name in BLOCK_TAGS:
            tag.insert_before(NavigableString("\n"))
            tag.insert_after(NavigableString("\n"))
    return soup.get_text("")


def html_to_text(
    html: str | bytes,
    drop_numeric_tables: bool = True,
    drop_hidden: bool = True,
    remove_artifacts: bool = True,
    reflow: bool = True,
) -> str:
    """Convert an HTML or inline-XBRL document to plain text (see module docstring)."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    html = _XML_DECL.sub("", html, count=1)
    html = _DATA_URI.sub(r'\1""', html)
    if not html.strip():
        return ""
    try:
        raw = _lxml_text(html, drop_numeric_tables, drop_hidden)
    except Exception:  # pragma: no cover - malformed documents
        raw = _bs4_text(html, drop_numeric_tables, drop_hidden)
    return postprocess(raw, remove_artifacts=remove_artifacts, reflow=reflow)


def normalize_text(text: str) -> str:
    """Collapse whitespace while keeping line structure (one block per line)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("­", "")  # soft hyphen
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def remove_page_artifacts(lines: list[str], min_repeats: int = 4, max_len: int = 90) -> list[str]:
    """Drop page numbers, contents links and repeated running headers or footers."""
    counts = Counter(line for line in lines if line)
    out = []
    for line in lines:
        if not line:
            out.append(line)
            continue
        if _PAGE_NUMBER.match(line) or _TOC_LINK.match(line):
            continue
        if (
            counts[line] >= min_repeats
            and len(line) <= max_len
            and not _TERMINAL.search(line)
            and not _PROTECTED_LINE.search(line)
        ):
            continue
        out.append(line)
    return out


def _last_word(line: str) -> str:
    m = re.search(r"([A-Za-z]+)[^A-Za-z]*$", line)
    return m.group(1).lower() if m else ""


def _is_all_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 2 and all(not c.islower() for c in letters)


def _continues(prev: str, nxt: str) -> bool:
    """True when *nxt* continues the sentence that *prev* leaves open."""
    if _TERMINAL.search(prev) or _ITEM_LINE.match(nxt) or _ITEM_LINE.match(prev) or _BULLET.match(nxt):
        return False
    if _is_all_caps(prev) or _is_all_caps(nxt):
        return False
    first = nxt[0]
    if first.islower():
        return True
    if prev.endswith((",", "-", "–", "—", "/", "&", "(")):
        return True
    if _last_word(prev) in _CONNECTORS and len(prev) >= 20:
        return first.isalnum() or first in "$€£¥(\"'"
    return False


def reflow_lines(lines: list[str]) -> list[str]:
    """Join lines that belong to the same sentence; blank lines between them are dropped."""
    out: list[str] = []
    pending_blank = False
    for line in lines:
        if not line:
            pending_blank = True
            continue
        if out and out[-1] and _continues(out[-1], line):
            prev = out[-1]
            if prev.endswith("-") and line[0].islower() and re.search(r"[A-Za-z]-$", prev):
                out[-1] = prev + line  # hyphenated line break: keep the hyphen ("machine-" "learning")
            else:
                out[-1] = prev + " " + line
            pending_blank = False
            continue
        if pending_blank and out:
            out.append("")
        out.append(line)
        pending_blank = False
    return out


def postprocess(text: str, remove_artifacts: bool = True, reflow: bool = True) -> str:
    """Normalise whitespace, remove page artifacts and reflow paragraphs."""
    text = normalize_text(text)
    lines = text.split("\n")
    if remove_artifacts:
        lines = remove_page_artifacts(lines)
    if reflow:
        lines = reflow_lines(lines)
    return normalize_text("\n".join(lines))


def pdf_to_text(data: bytes | str | Path, remove_artifacts: bool = True, reflow: bool = True) -> str:
    """Extract text from a PDF (requires the optional ``pypdf`` dependency)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PDF support requires 'pip install aidisclose[pdf]'") from exc
    import io

    reader = PdfReader(str(data)) if isinstance(data, (str, Path)) else PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return postprocess("\n\n".join(pages), remove_artifacts=remove_artifacts, reflow=reflow)


def _looks_like_html(head: str) -> bool:
    head = head.lstrip()[:2000].lower()
    return head.startswith("<!doctype html") or "<html" in head or "<body" in head or "<?xml" in head or "<div" in head


def sgml_text_to_text(text: str, remove_artifacts: bool = True, reflow: bool = True) -> str:
    """Strip SGML/EDGAR wrapper tags from legacy plain-text filings."""
    text = re.sub(r"<[^>\n]{1,200}>", " ", text)
    return postprocess(text, remove_artifacts=remove_artifacts, reflow=reflow)


def extract_text(data: bytes | str, filename: str = "", content_type: str = "", reflow: bool = True) -> str:
    """Dispatch to the right extractor based on file name, content type or content."""
    name = filename.lower()
    ctype = content_type.lower()
    if name.endswith(".pdf") or "pdf" in ctype or (isinstance(data, bytes) and data[:5] == b"%PDF-"):
        return pdf_to_text(data if isinstance(data, bytes) else data.encode(), reflow=reflow)
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if name.endswith((".htm", ".html", ".xhtml", ".xml")) or "html" in ctype or _looks_like_html(text):
        return html_to_text(text, reflow=reflow)
    return sgml_text_to_text(text, reflow=reflow)
