"""Lightweight, dependency-free language identification.

The bundled rule cues and lexicons support English text. Reports from multilingual
sources (e.g. ESEF filings) are screened with a stop-word profile: the share of
tokens that are function words of each candidate language. English text
typically scores 0.35-0.50 on the English profile and far below that on the
others.
"""

from __future__ import annotations

import re

_PROFILES = {
    "en": "the of and to in a is that for on with as by are be this from or at it an which our we have has were not will its their been".split(),
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei".split(),
    "fr": "de la le et les des en un une du est que pour dans qui par sur au avec ce sont plus pas ou ses leur aux été".split(),
    "es": "de la que el en y los del se las por un para con una su al lo como más pero sus le ha".split(),
    "it": "di e il la che in un per del della è con non una le si dei sono alla al da gli nel".split(),
    "nl": "de en van het een in is op te dat voor met zijn niet aan er als ook door bij worden wordt".split(),
    "sv": "och i att det som en på är av för med till den har de inte om ett vi var sig".split(),
    "da": "og i at det en til er som på de med af for ikke der var sig har et vi".split(),
    "fi": "ja on ei se että oli hän kuin mutta tai ovat myös sen joka ole sekä".split(),
    "pt": "de a o que e do da em um para é com não uma os no se na por mais as dos".split(),
    "pl": "i w na z że do nie się jest to o jak od po za oraz dla przez".split(),
}
_PROFILE_SETS = {k: set(v) for k, v in _PROFILES.items()}
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def language_scores(text: str, max_tokens: int = 20000) -> dict[str, float]:
    tokens = [t.lower() for t in _TOKEN.findall(text[: max_tokens * 8])][:max_tokens]
    if not tokens:
        return {k: 0.0 for k in _PROFILES}
    n = len(tokens)
    return {lang: sum(1 for t in tokens if t in words) / n for lang, words in _PROFILE_SETS.items()}


def detect_language(text: str, min_tokens: int = 30, min_score: float = 0.15) -> tuple[str | None, dict[str, float]]:
    """Return ``(language_code_or_None, scores)``."""
    scores = language_scores(text)
    if len(_TOKEN.findall(text[:5000])) < min_tokens:
        return None, scores
    best = max(scores, key=scores.get)
    if scores[best] < min_score:
        return None, scores
    return best, scores


def is_english(text: str) -> bool:
    lang, _ = detect_language(text)
    return lang == "en"
