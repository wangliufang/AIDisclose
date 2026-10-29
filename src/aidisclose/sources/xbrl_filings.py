"""Adapter for ESEF (EU) and UKSEF (UK) financial reports via filings.xbrl.org.

filings.xbrl.org, operated by XBRL International, indexes inline-XBRL
reports filed under the European Single Electronic Format (ESEF) and its UK
counterpart, and exposes them through a JSON:API interface. Reports are
XHTML documents in the language chosen by the issuer. Language is screened
from the file name and, after download, from the text itself; by default only
English-language reports are measured. The index may also contain interim
reports, and the period end alone does not identify annual reports. Reports are split into
parts (``AR:management``, ``AR:risk``, ``AR:sustainability``, ...).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .._version import __version__
from ..models import Document, Filing
from .base import HttpClient, Source, fiscal_year_from_period_end

BASE_URL = "https://filings.xbrl.org"
API_URL = BASE_URL + "/api/filings"
ENTITY_FILINGS_URL = BASE_URL + "/api/entities/{lei}/filings"

_LANG_IN_NAME = re.compile(r"[-_.]([a-z]{2})(?:[-_.](?:\d+|[a-z]{2}))?\.x?html?$", re.I)
_EU_LANGS = {
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "is",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
}


def language_hint(report_url: str | None) -> str | None:
    """Guess the report language from its file name (e.g. ``...-2023-12-31-en.xhtml``)."""
    if not report_url:
        return None
    m = _LANG_IN_NAME.search(report_url)
    if m and m.group(1).lower() in _EU_LANGS:
        return m.group(1).lower()
    return None


class XbrlFilingsSource(Source):
    """List and download ESEF/UKSEF reports from filings.xbrl.org."""

    name = "xbrl"

    def __init__(
        self,
        user_agent: str = f"aidisclose/{__version__} (research software; set a contact address)",
        cache_dir: str | Path | None = ".aidisclose_cache/xbrl",
        max_rps: float = 2.0,
        session=None,
    ) -> None:
        self.http = HttpClient(user_agent, max_rps=max_rps, cache_dir=cache_dir, session=session)

    # -------------------------------------------------------------- discovery
    @staticmethod
    def build_filter(countries: Iterable[str] | None, start_year: int | None, end_year: int | None) -> list[dict]:
        flt: list[dict[str, Any]] = []
        if countries:
            flt.append({"name": "country", "op": "in", "val": [c.upper() for c in countries]})
        # Fiscal years ending January-May belong to the previous year, so widen the period window.
        if start_year is not None:
            flt.append({"name": "period_end", "op": "ge", "val": f"{start_year}-06-01"})
        if end_year is not None:
            flt.append({"name": "period_end", "op": "le", "val": f"{end_year + 1}-05-31"})
        return flt

    def _pages(self, url: str, params: dict[str, Any] | None, max_pages: int | None):
        page = 0
        next_url: str | None = url
        next_params = params
        while next_url:
            data = self.http.get_json(next_url, params=next_params, use_cache=False)
            yield data
            page += 1
            if max_pages is not None and page >= max_pages:
                return
            nxt = (data.get("links") or {}).get("next")
            if nxt and nxt.startswith("/"):
                nxt = BASE_URL + nxt
            next_url, next_params = nxt, None

    def list_filings(
        self,
        countries: Iterable[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        leis: Iterable[str] | None = None,
        english_only: bool = True,
        page_size: int = 200,
        max_pages: int | None = None,
        keep_all_versions: bool = False,
        languages: Iterable[str] | None = None,
    ) -> list[Filing]:
        """List reports by country and fiscal year, or by LEI.

        With ``english_only`` (default) reports whose file name indicates
        another language are dropped; reports without a language hint are kept
        and screened after download. ``languages``, when supplied, selects
        acceptable hints and takes precedence over ``english_only``; its order
        determines language preference during deduplication. Unless
        ``keep_all_versions`` is set, one report per entity and valid period
        is kept. Unknown periods are not merged across different records.
        This listing does not verify whether a report is annual or interim.
        """
        countries = tuple(c.upper() for c in countries) if countries is not None else ()
        languages = (
            tuple(dict.fromkeys(str(lang).strip().lower() for lang in languages)) if languages is not None else None
        )
        if languages is not None and (not languages or any(not lang for lang in languages)):
            raise ValueError("languages must contain nonempty language codes")
        params_base = {"include": "entity", "page[size]": page_size}
        requests_ = []
        if leis:
            for lei in leis:
                requests_.append((ENTITY_FILINGS_URL.format(lei=lei), dict(params_base)))
        else:
            params = dict(params_base)
            flt = self.build_filter(countries, start_year, end_year)
            if flt:
                params["filter"] = json.dumps(flt)
            requests_.append((API_URL, params))

        filings: list[Filing] = []
        for url, params in requests_:
            for data in self._pages(url, params, max_pages):
                entities = {
                    item["id"]: item.get("attributes", {})
                    for item in data.get("included", []) or []
                    if item.get("type") == "entity"
                }
                for item in data.get("data", []) or []:
                    f = self._to_filing(item, entities)
                    if f is None:
                        continue
                    if start_year is not None and (f.fiscal_year is None or f.fiscal_year < start_year):
                        continue
                    if end_year is not None and (f.fiscal_year is None or f.fiscal_year > end_year):
                        continue
                    if countries and (f.country or "").upper() not in countries:
                        continue
                    allowed = languages if languages is not None else (("en",) if english_only else None)
                    if allowed is not None and f.language is not None and f.language not in allowed:
                        continue
                    filings.append(f)
        unique = {}
        for filing in filings:
            previous = unique.get(filing.filing_id)
            if previous is not None and previous.to_dict() != filing.to_dict():
                raise ValueError(f"conflicting records for filing identifier {filing.filing_id}")
            unique[filing.filing_id] = filing
        filings = list(unique.values())
        if not keep_all_versions:
            filings = self._dedupe(filings, languages=languages)
        filings.sort(key=lambda f: (f.firm_id, f.fiscal_year or 0))
        return filings

    def _to_filing(self, item: dict[str, Any], entities: dict[str, dict[str, Any]]) -> Filing | None:
        attr = item.get("attributes", {}) or {}
        report_url = attr.get("report_url")
        if not report_url:
            return None
        rel = ((item.get("relationships") or {}).get("entity") or {}).get("data") or {}
        ent = entities.get(rel.get("id"), {})
        fxo = str(attr.get("fxo_id") or item.get("id"))
        record_id = str(item.get("id") or "")
        identity = record_id or "url-" + hashlib.sha256(report_url.encode("utf-8")).hexdigest()[:20]
        lei = str(ent.get("identifier") or "").strip().upper()
        identity_basis = "entity_identifier"
        match = re.match(r"^([A-Za-z0-9]{20})-\d{4}-\d{2}-\d{2}-", fxo)
        if lei and match and lei.upper() != match.group(1).upper():
            raise ValueError(f"XBRL record {record_id or identity!r} has conflicting entity identifiers")
        if not lei:
            if not match:
                raise ValueError(f"XBRL record {record_id or identity!r} lacks a usable entity identifier")
            lei = match.group(1).upper()
            identity_basis = "fxo_id_prefix_unverified"
        period = attr.get("period_end")
        form = "UKSEF" if "-UKSEF-" in fxo.upper() else ("ESEF" if "-ESEF-" in fxo.upper() else "iXBRL")
        return Filing(
            source=self.name,
            filing_id=f"xbrl:{fxo}:{identity}",
            firm_id=f"lei:{lei}",
            firm_name=ent.get("name") or lei,
            fiscal_year=fiscal_year_from_period_end(period),
            period_end=period,
            country=(attr.get("country") or "").upper() or None,
            form=form,
            url=BASE_URL + report_url if report_url.startswith("/") else report_url,
            language=language_hint(report_url),
            extra={
                "lei": lei,
                "entity_identity_basis": identity_basis,
                "fxo_id": fxo,
                "source_record_id": record_id or None,
                "processed": attr.get("processed"),
                "date_added": attr.get("date_added"),
                "package_url": attr.get("package_url"),
                "error_count": attr.get("error_count"),
            },
        )

    @staticmethod
    def _dedupe(filings: list[Filing], languages: Iterable[str] | None = None) -> list[Filing]:
        best: dict[tuple[str, str | None], Filing] = {}
        preferred = tuple(languages) if languages is not None else ("en",)

        def rank(f):
            return (
                f.language in preferred,
                -preferred.index(f.language) if f.language in preferred else 0,
                str(f.extra.get("processed") or ""),
            )

        for f in filings:
            period = f.period_end if fiscal_year_from_period_end(f.period_end) is not None else None
            key = (f.firm_id, period if period is not None else f"unknown:{f.filing_id}")
            cur = best.get(key)
            if cur is None or rank(f) > rank(cur):
                best[key] = f
        return list(best.values())

    # --------------------------------------------------------------- download
    def fetch_document(self, filing: Filing) -> Document:
        if not filing.url:
            raise ValueError(f"filing {filing.filing_id} has no URL")
        data = self.http.get_bytes(filing.url)
        return self.make_document(filing, data, filename=filing.url)
