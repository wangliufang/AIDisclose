"""SEC EDGAR adapter for annual reports: Forms 10-K, 20-F and 40-F.

Uses the public EDGAR submissions API (``data.sec.gov/submissions``) to list
filings and downloads documents from the EDGAR archive. The SEC's fair-access
policy requires a declared User-Agent with contact information and limits
automated access to 10 requests per second; this adapter enforces both
(default 8 requests per second).

Annual reports of foreign private issuers (Form 20-F) and of Canadian issuers
under the multijurisdictional disclosure system (Form 40-F) are supported, so
that listed firms from many countries can be measured with one adapter.

Exhibits
--------
Some 10-K filings incorporate Items (often Item 7, MD&A) by reference to the
annual report to shareholders filed as Exhibit 13, and the substance of a
40-F is in its exhibits (Annual Information Form and MD&A). With
``exhibits="auto"`` (default) the adapter reads the filing index and adds

* ``EX-13`` to a 10-K when the 10-K says that content is incorporated by
  reference to Exhibit 13 or to the annual report to shareholders, and
* ``EX-99.*`` exhibits described as the Annual Information Form or MD&A to a
  40-F (if no exhibit is described that way, every ``EX-99.*`` exhibit whose
  description does not name financial statements, consents, certifications
  or press releases).

Exhibits become sections named ``10-K:EX-13``, ``40-F:AIF``, ``40-F:MDA``
or ``40-F:EX-99.x``; the 40-F cover document is ``40-F:main``. The country
of a filer is taken from its EDGAR business-address code and mapped to ISO
3166-1 alpha-2 (:func:`country_from_edgar`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..industry import sic_to_nace_section
from ..models import Document, Filing
from ..text.extract import extract_text
from ..text.language import detect_language
from ..text.sections import split_sections
from .base import HttpClient, Source, fiscal_year_from_period_end

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}-index.htm"
SEC_ROOT = "https://www.sec.gov"

# The 50 states, DC and "X1" (United States). Puerto Rico ("PR"), Guam ("GU")
# and the U.S. Virgin Islands ("VI") map to their own ISO codes.
US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH "
    "OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY X1".split()
)
# EDGAR state and country codes -> ISO 3166-1 alpha-2, transcribed from the SEC
# table "EDGAR State and Country Codes" (accessed 2026-10-23). "XX" (unknown)
# maps to None; "P8" (Netherlands Antilles) keeps the withdrawn ISO code "AN".
EDGAR_COUNTRY = {
    "A0": "CA",
    "A1": "CA",
    "A2": "CA",
    "A3": "CA",
    "A4": "CA",
    "A5": "CA",
    "A6": "CA",
    "A7": "CA",
    "A8": "CA",
    "A9": "CA",
    "B0": "CA",
    "Z4": "CA",
    "B2": "AF",
    "Y6": "AX",
    "B3": "AL",
    "B4": "DZ",
    "B5": "AS",
    "B6": "AD",
    "B7": "AO",
    "1A": "AI",
    "B8": "AQ",
    "B9": "AG",
    "C1": "AR",
    "1B": "AM",
    "1C": "AW",
    "C3": "AU",
    "C4": "AT",
    "1D": "AZ",
    "C5": "BS",
    "C6": "BH",
    "C7": "BD",
    "C8": "BB",
    "1F": "BY",
    "C9": "BE",
    "D1": "BZ",
    "G6": "BJ",
    "D0": "BM",
    "D2": "BT",
    "D3": "BO",
    "1E": "BA",
    "B1": "BW",
    "D4": "BV",
    "D5": "BR",
    "D6": "IO",
    "D9": "BN",
    "E0": "BG",
    "X2": "BF",
    "E2": "BI",
    "E3": "KH",
    "E4": "CM",
    "E8": "CV",
    "E9": "KY",
    "F0": "CF",
    "F2": "TD",
    "F3": "CL",
    "F4": "CN",
    "F6": "CX",
    "F7": "CC",
    "F8": "CO",
    "F9": "KM",
    "G0": "CG",
    "Y3": "CD",
    "G1": "CK",
    "G2": "CR",
    "L7": "CI",
    "1M": "HR",
    "G3": "CU",
    "G4": "CY",
    "2N": "CZ",
    "G7": "DK",
    "1G": "DJ",
    "G9": "DM",
    "G8": "DO",
    "H1": "EC",
    "H2": "EG",
    "H3": "SV",
    "H4": "GQ",
    "1J": "ER",
    "1H": "EE",
    "H5": "ET",
    "H7": "FK",
    "H6": "FO",
    "H8": "FJ",
    "H9": "FI",
    "I0": "FR",
    "I3": "GF",
    "I4": "PF",
    "2C": "TF",
    "I5": "GA",
    "I6": "GM",
    "2Q": "GE",
    "2M": "DE",
    "J0": "GH",
    "J1": "GI",
    "J3": "GR",
    "J4": "GL",
    "J5": "GD",
    "J6": "GP",
    "GU": "GU",
    "J8": "GT",
    "Y7": "GG",
    "J9": "GN",
    "S0": "GW",
    "K0": "GY",
    "K1": "HT",
    "K4": "HM",
    "X4": "VA",
    "K2": "HN",
    "K3": "HK",
    "K5": "HU",
    "K6": "IS",
    "K7": "IN",
    "K8": "ID",
    "K9": "IR",
    "L0": "IQ",
    "L2": "IE",
    "Y8": "IM",
    "L3": "IL",
    "L6": "IT",
    "L8": "JM",
    "M0": "JP",
    "Y9": "JE",
    "M2": "JO",
    "1P": "KZ",
    "M3": "KE",
    "J2": "KI",
    "M4": "KP",
    "M5": "KR",
    "M6": "KW",
    "1N": "KG",
    "M7": "LA",
    "1R": "LV",
    "M8": "LB",
    "M9": "LS",
    "N0": "LR",
    "N1": "LY",
    "N2": "LI",
    "1Q": "LT",
    "N4": "LU",
    "N5": "MO",
    "1U": "MK",
    "N6": "MG",
    "N7": "MW",
    "N8": "MY",
    "N9": "MV",
    "O0": "ML",
    "O1": "MT",
    "1T": "MH",
    "O2": "MQ",
    "O3": "MR",
    "O4": "MU",
    "2P": "YT",
    "O5": "MX",
    "1K": "FM",
    "1S": "MD",
    "O9": "MC",
    "P0": "MN",
    "Z5": "ME",
    "P1": "MS",
    "P2": "MA",
    "P3": "MZ",
    "E1": "MM",
    "T6": "NA",
    "P5": "NR",
    "P6": "NP",
    "P7": "NL",
    "P8": "AN",
    "1W": "NC",
    "Q2": "NZ",
    "Q3": "NI",
    "Q4": "NE",
    "Q5": "NG",
    "Q6": "NU",
    "Q7": "NF",
    "1V": "MP",
    "Q8": "NO",
    "P4": "OM",
    "R0": "PK",
    "1Y": "PW",
    "1X": "PS",
    "R1": "PA",
    "R2": "PG",
    "R4": "PY",
    "R5": "PE",
    "R6": "PH",
    "R8": "PN",
    "R9": "PL",
    "S1": "PT",
    "PR": "PR",
    "S3": "QA",
    "S4": "RE",
    "S5": "RO",
    "1Z": "RU",
    "S6": "RW",
    "Z0": "BL",
    "U8": "SH",
    "U7": "KN",
    "U9": "LC",
    "Z1": "MF",
    "V0": "PM",
    "V1": "VC",
    "Y0": "WS",
    "S8": "SM",
    "S9": "ST",
    "T0": "SA",
    "T1": "SN",
    "Z2": "RS",
    "T2": "SC",
    "T8": "SL",
    "U0": "SG",
    "2B": "SK",
    "2A": "SI",
    "D7": "SB",
    "U1": "SO",
    "T3": "ZA",
    "1L": "GS",
    "U3": "ES",
    "F1": "LK",
    "V2": "SD",
    "V3": "SR",
    "L9": "SJ",
    "V6": "SZ",
    "V7": "SE",
    "V8": "CH",
    "V9": "SY",
    "F5": "TW",
    "2D": "TJ",
    "W0": "TZ",
    "W1": "TH",
    "Z3": "TL",
    "W2": "TG",
    "W3": "TK",
    "W4": "TO",
    "W5": "TT",
    "W6": "TN",
    "W8": "TR",
    "2E": "TM",
    "W7": "TC",
    "2G": "TV",
    "W9": "UG",
    "2H": "UA",
    "C0": "AE",
    "X0": "GB",
    "2J": "UM",
    "X3": "UY",
    "2K": "UZ",
    "2L": "VU",
    "X5": "VE",
    "Q1": "VN",
    "D8": "VG",
    "VI": "VI",
    "X8": "WF",
    "U5": "EH",
    "T7": "YE",
    "Y4": "ZM",
    "Y5": "ZW",
}
_UA_RX = re.compile(r"\S+@\S+\.\S+")
_BY_REFERENCE = re.compile(
    r"incorporated\s+(?:herein\s+)?by\s+reference[^.]{0,300}?(?:exhibit\s+13\b|annual\s+report\s+to\s+(?:share|stock)holders)"
    r"|(?:exhibit\s+13\b|annual\s+report\s+to\s+(?:share|stock)holders)[^.]{0,300}?incorporated\s+(?:herein\s+)?by\s+reference",
    re.IGNORECASE,
)
_AIF = re.compile(r"annual\s+information\s+form|\baif\b", re.I)
_NOT_NARRATIVE = re.compile(r"financial\s+statements|consent|certif|press\s+release|news\s+release", re.I)
_MDA = re.compile(r"management'?s?\s+discussion|md\s*&\s*a|\bmda\b", re.I)

ANNUAL_FORMS = ("10-K", "10-K405", "10-KT", "20-F", "40-F")


def country_from_edgar(code: str | None) -> str | None:
    """Map an EDGAR state/country code to an ISO 3166-1 alpha-2 country code."""
    if not code:
        return None
    code = code.upper()
    if code in US_STATES:
        return "US"
    return EDGAR_COUNTRY.get(code)


def parse_filing_index(html: str | bytes) -> list[dict[str, str]]:
    """Parse an EDGAR filing index page into ``[{seq, description, document, type, url}]``."""
    import lxml.html

    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    root = lxml.html.document_fromstring(html)
    rows: list[dict[str, str]] = []
    for table in root.iter("table"):
        trs = table.findall(".//tr")
        if not trs:
            continue
        header = [" ".join(c.text_content().split()).lower() for c in trs[0].findall("./th")]
        if header[:5] != ["seq", "description", "document", "type", "size"]:
            continue
        for tr in trs[1:]:
            cells = tr.findall("./td")
            if len(cells) < 4:
                continue
            link = cells[2].find(".//a")
            href = link.get("href") if link is not None else ""
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc=") :]
            url = SEC_ROOT + href if href.startswith("/") else href
            rows.append(
                {
                    "seq": " ".join(cells[0].text_content().split()),
                    "description": " ".join(cells[1].text_content().replace("\u2019", "'").split()),
                    "document": " ".join((link.text_content() if link is not None else "").split()),
                    "type": " ".join(cells[3].text_content().split()).upper(),
                    "url": url,
                }
            )
    return rows


class EdgarSource(Source):
    """List and download annual reports (10-K, 20-F, 40-F) from SEC EDGAR.

    Parameters
    ----------
    user_agent:
        Required by the SEC, e.g. ``"Jane Doe jane.doe@university.edu"``.
    cache_dir:
        Directory for cached API responses and documents.
    max_rps:
        Request rate limit (must stay below the SEC maximum of 10).
    exhibits:
        ``"auto"`` (default; see module docstring), ``"none"`` or ``"all"``
        (always add EX-13 to 10-K filings and all EX-99.* to 40-F filings).
    """

    name = "edgar"

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path | None = ".aidisclose_cache/edgar",
        max_rps: float = 8.0,
        session=None,
        exhibits: str = "auto",
        fiscal_year_rule: str = "jan_may_previous",
    ):
        if not user_agent or not _UA_RX.search(user_agent):
            raise ValueError(
                "SEC EDGAR requires a User-Agent with contact details, e.g. 'Jane Doe jane.doe@university.edu'"
            )
        if not math.isfinite(max_rps) or not 0 < max_rps <= 10:
            raise ValueError("SEC request rate must be finite and greater than zero, at most 10 per second")
        if exhibits not in {"auto", "none", "all"}:
            raise ValueError("exhibits must be 'auto', 'none' or 'all'")
        if fiscal_year_rule not in {"period_end_year", "jan_may_previous"}:
            raise ValueError("fiscal_year_rule must be 'period_end_year' or 'jan_may_previous'")
        self.http = HttpClient(user_agent, max_rps=max_rps, cache_dir=cache_dir, session=session)
        self.exhibits = exhibits
        self.fiscal_year_rule = fiscal_year_rule

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_rps": self.http.min_interval and round(1 / self.http.min_interval, 6),
            "exhibits": self.exhibits,
            "fiscal_year_rule": self.fiscal_year_rule,
            "user_agent_contact_present": bool(self.http.user_agent),
        }

    # -------------------------------------------------------------- discovery
    def lookup_ciks(self, tickers: Iterable[str]) -> dict[str, int]:
        """Map ticker symbols to CIK numbers using the SEC ticker file."""
        data = self.http.get_json(TICKERS_URL)
        table = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}
        out = {}
        for t in tickers:
            cik = table.get(t.upper())
            if cik is None:
                raise KeyError(f"ticker {t!r} not found in the SEC ticker file")
            out[t.upper()] = cik
        return out

    def _submission_tables(self, cik: int, start_year: int | None) -> tuple[dict[str, Any], list[dict[str, list]]]:
        sub = self.http.get_json(SUBMISSIONS_URL.format(cik=cik), use_cache=False)
        tables = [sub.get("filings", {}).get("recent", {})]
        for f in sub.get("filings", {}).get("files", []) or []:
            filing_to = str(f.get("filingTo", ""))
            if start_year is not None and filing_to[:4].isdigit() and int(filing_to[:4]) < start_year:
                continue
            tables.append(self.http.get_json(SUBMISSIONS_FILE_URL.format(name=f["name"])))
        return sub, tables

    def list_filings(
        self,
        ciks: Iterable[int | str] | None = None,
        tickers: Iterable[str] | None = None,
        forms: Iterable[str] = ("10-K",),
        start_year: int | None = None,
        end_year: int | None = None,
        include_amendments: bool = False,
    ) -> list[Filing]:
        """List annual reports for the given companies.

        ``forms`` may include ``"10-K"``, ``"20-F"`` and ``"40-F"``.
        ``start_year``/``end_year`` filter on the fiscal year (derived from the
        reporting period; periods ending January-May count as the previous
        fiscal year). When the period is absent, a valid filing date may be
        used for an explicitly marked filing-year-minus-one estimate. An
        invalid supplied period remains unknown. Amendments (``.../A``) are
        excluded unless requested.
        """
        cik_list = [int(c) for c in (ciks or [])]
        if tickers:
            cik_list += list(self.lookup_ciks(tickers).values())
        if not cik_list:
            raise ValueError("give at least one CIK or ticker")
        wanted = {f.upper() for f in forms}
        if include_amendments:
            wanted |= {f + "/A" for f in list(wanted)}
        out: list[Filing] = []
        for cik in dict.fromkeys(cik_list):
            sub, tables = self._submission_tables(cik, start_year)
            name = sub.get("name", f"CIK {cik}")
            sic = str(sub.get("sic") or "") or None
            addr = (sub.get("addresses") or {}).get("business") or {}
            loc = str(addr.get("stateOrCountry") or "")
            country = country_from_edgar(loc)
            seen: set[str] = set()
            for t in tables:
                n = len(t.get("accessionNumber", []))
                for i in range(n):
                    form = str(t["form"][i]).upper()
                    if form not in wanted:
                        continue
                    acc = t["accessionNumber"][i]
                    if acc in seen:
                        continue
                    seen.add(acc)
                    period = (t.get("reportDate") or [""] * n)[i] or ""
                    filed = (t.get("filingDate") or [""] * n)[i] or ""
                    fy = fiscal_year_from_period_end(period, self.fiscal_year_rule) if period else None
                    basis = "period_end" if fy is not None else ("invalid_period_end" if period else "missing")
                    # An invalid supplied reporting period is not the same as
                    # an absent period. Never replace an invalid date silently.
                    if not period and filed:
                        filing_year = fiscal_year_from_period_end(filed, convention="period_end_year")
                        if filing_year is not None:
                            fy = filing_year - 1
                            basis = "filing_year_minus_one_estimate"
                    if start_year is not None and (fy is None or fy < start_year):
                        continue
                    if end_year is not None and (fy is None or fy > end_year):
                        continue
                    doc = t["primaryDocument"][i]
                    acc_nodash = acc.replace("-", "")
                    out.append(
                        Filing(
                            source=self.name,
                            filing_id=f"edgar:{cik}:{acc}",
                            firm_id=f"cik:{cik}",
                            firm_name=name,
                            fiscal_year=fy,
                            period_end=period or None,
                            country=country,
                            industry_code=sic,
                            industry_scheme="SIC" if sic else None,
                            form=form,
                            url=ARCHIVE_URL.format(cik=cik, acc=acc_nodash, doc=doc),
                            language="en",
                            extra={
                                "filing_date": filed,
                                "fiscal_year_rule": self.fiscal_year_rule,
                                "fiscal_year_basis": basis,
                                "accession": acc,
                                "cik": cik,
                                "state_or_country": loc,
                                "sic_description": sub.get("sicDescription"),
                                "nace_section": sic_to_nace_section(sic),
                                "tickers": sub.get("tickers"),
                                "index_url": INDEX_URL.format(cik=cik, acc_nodash=acc_nodash, acc=acc),
                            },
                        )
                    )
        out.sort(key=lambda f: (f.firm_id, f.fiscal_year or 0, f.extra.get("filing_date", "")))
        return out

    # --------------------------------------------------------------- exhibits
    def filing_index(self, filing: Filing) -> list[dict[str, str]]:
        url = filing.extra.get("index_url")
        acc, cik = filing.extra.get("accession"), filing.extra.get("cik")
        if not url and acc and cik:  # filing lists written by aidisclose 0.1
            url = INDEX_URL.format(cik=int(cik), acc_nodash=str(acc).replace("-", ""), acc=acc)
        if not url:
            return []
        return parse_filing_index(self.http.get_bytes(url))

    def _select_exhibits(self, filing: Filing, primary_text: str) -> list[tuple[str, dict[str, str]]]:
        form = (filing.form or "").upper().replace("/A", "")
        if self.exhibits == "none":
            return []
        if form.startswith("10-K"):
            if self.exhibits == "auto" and not _BY_REFERENCE.search(primary_text):
                return []
            rows = [r for r in self.filing_index(filing) if r["type"].startswith("EX-13")]
            return [("10-K:EX-13", r) for r in rows[:1]]
        if form.startswith("40-F"):
            rows = [r for r in self.filing_index(filing) if r["type"].startswith("EX-99")]
            out = []
            for r in rows:
                desc = f"{r['description']} {r['document']}"
                if _AIF.search(desc):
                    out.append(("40-F:AIF", r))
                elif _MDA.search(desc):
                    out.append(("40-F:MDA", r))
                elif self.exhibits == "all":
                    out.append((f"40-F:{r['type']}", r))
            if not out:  # exhibits not described: take the narrative-looking ones
                out = [(f"40-F:{r['type']}", r) for r in rows if not _NOT_NARRATIVE.search(r["description"])]
            return out
        return []

    # --------------------------------------------------------------- download
    def fetch_document(self, filing: Filing) -> Document:
        if not filing.url:
            raise ValueError(f"filing {filing.filing_id} has no URL")
        text = extract_text(self.http.get_bytes(filing.url), filename=filing.url)
        sections = split_sections(text, filing.form or "10-K")
        exhibits = self._select_exhibits(filing, text)
        if list(sections) == ["full"]:
            sections = {f"{schema_label(filing.form)}:main": text} if exhibits else {}
        attached = []
        for label, row in exhibits:
            ex_text = extract_text(self.http.get_bytes(row["url"]), filename=row["url"])
            if not ex_text.strip():
                continue
            key, n = label, 2
            while key in sections:
                key, n = f"{label}({n})", n + 1
            sections[key] = ex_text
            attached.append({"section": key, "type": row["type"], "url": row["url"]})
        if attached:
            filing.extra["exhibits"] = attached
        full = "\n\n".join(sections.values()) if sections else text
        lang, scores = detect_language(full)
        return Document(filing=filing, text=full, sections=sections, language=lang, language_scores=scores)


def schema_label(form: str | None) -> str:
    """Section prefix of a form: ``10-K405`` and ``10-K/A`` -> ``10-K``."""
    f = (form or "DOC").upper().replace("/A", "")
    for prefix in ("10-K", "20-F", "40-F"):
        if f.startswith(prefix):
            return prefix
    return f
