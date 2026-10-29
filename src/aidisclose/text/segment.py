"""Sentence segmentation tuned for English corporate disclosures.

The splitter is deterministic and dependency-free. It treats every line of the
extracted text as a block (block-level HTML elements become lines), never joins
text across blocks, and splits blocks at sentence-final punctuation unless the
preceding token is a known abbreviation (``Inc.``, ``U.S.``, ``e.g.``, ...), an
initial or a section number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ABBREVIATIONS = {
    "inc",
    "corp",
    "co",
    "ltd",
    "llc",
    "plc",
    "lp",
    "llp",
    "no",
    "nos",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "cf",
    "approx",
    "u.s",
    "u.k",
    "e.u",
    "u.n",
    "mr",
    "mrs",
    "ms",
    "dr",
    "jr",
    "sr",
    "st",
    "fig",
    "figs",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "est",
    "dept",
    "div",
    "gov",
    "intl",
    "mfg",
    "n.a",
    "s.a",
    "a.g",
    "n.v",
    "b.v",
    "s.p.a",
    "a.s",
    "a/s",
    "ab",
    "oy",
    "oyj",
    "sec",
    "art",
    "para",
    "ref",
    "vol",
    "pp",
    "p",
    "ph.d",
    "m.d",
    "a.i",
    "nasdaq",
    "avg",
    "incl",
    "excl",
    "max",
    "min",
    "resp",
    "viz",
    "al",
}

_BOUNDARY = re.compile(r"([.!?])([\"')\]]*)(\s+)(?=[\"'(\[]?[A-Z0-9••])")
_TOKEN_BEFORE = re.compile(r"([A-Za-z][A-Za-z./&]*)$")
_NUMBERED = re.compile(r"(?:^|\s)(?:item|note|section|part|article|rule|no)\s*\d+[a-z]?$", re.I)
_SEMICOLON_SPLIT = re.compile(r";\s+")


@dataclass(frozen=True)
class Span:
    block: int
    start: int
    end: int
    text: str


def _is_abbreviation(prefix: str) -> bool:
    m = _TOKEN_BEFORE.search(prefix)
    if not m:
        return False
    token = m.group(1).lower().rstrip(".")
    if token in ABBREVIATIONS:
        return True
    # single initials such as "J." or "A.B."
    if re.fullmatch(r"[a-z](?:\.[a-z])*", token):
        return True
    return False


def split_block(block: str, max_chars: int = 1200) -> list[str]:
    """Split one block of text into sentences."""
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(block):
        end = m.end(2)
        prefix = block[start : m.start(1)]
        if m.group(1) == "." and (_is_abbreviation(prefix) or _NUMBERED.search(prefix)):
            continue
        piece = block[start:end].strip()
        if piece:
            sentences.append(piece)
        start = m.end(3)
    tail = block[start:].strip()
    if tail:
        sentences.append(tail)
    out: list[str] = []
    for s in sentences:
        if len(s) > max_chars:
            out.extend(p.strip() for p in _SEMICOLON_SPLIT.split(s) if p.strip())
        else:
            out.append(s)
    return out


def split_sentences(text: str, max_chars: int = 1200) -> list[Span]:
    """Split text into sentences, never crossing line (block) boundaries."""
    spans: list[Span] = []
    pos = 0
    for b, block in enumerate(text.split("\n")):
        if block.strip():
            cursor = 0
            for sent in split_block(block, max_chars=max_chars):
                idx = block.find(sent, cursor)
                if idx < 0:
                    idx = cursor
                spans.append(Span(b, pos + idx, pos + idx + len(sent), sent))
                cursor = idx + len(sent)
        pos += len(block) + 1
    return spans
