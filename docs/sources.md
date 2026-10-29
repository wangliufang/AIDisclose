# Sources and adapters

AIDisclose separates *where documents come from* (sources) from *how they are
measured* (the pipeline). All sources return the same `Filing` metadata
(firm identifier, fiscal year, ISO country code, industry code, URL) and the
same `Document` object (plain text, sections, and language).

| Source | Coverage | Forms | Firm id | Country |
|---|---|---|---|---|
| `edgar` | companies filing with the U.S. SEC, including foreign private issuers | 10-K, 20-F, 40-F (+ exhibits) | `cik:<CIK>` | from the EDGAR business-address code |
| `xbrl` / `esef` | ESEF (EU/EEA) and UKSEF reports indexed by filings.xbrl.org | inline-XBRL financial reports | `lei:<LEI>` | from the index |
| `local` | any folder of HTML, XHTML, PDF or text reports | any | metadata CSV or file name | metadata CSV |

## SEC EDGAR

```python
src = aid.EdgarSource(user_agent="Jane Doe jane.doe@university.edu")
us = src.list_filings(tickers=["MSFT", "JPM"], start_year=2018, end_year=2025)               # 10-K
fpi = src.list_filings(tickers=["SAP", "TM"], forms=("20-F",), start_year=2020)            # foreign issuers
ca = src.list_filings(tickers=["CNI"], forms=("40-F",), start_year=2020)                   # Canadian MJDS filers
```

* Filings are listed from the public submissions API
  (`https://data.sec.gov/submissions/CIK##########.json`), including the
  additional history files for older filings, and documents are downloaded
  from the EDGAR archive.
* The SEC's fair-access policy requires a declared User-Agent with contact
  details and allows at most 10 requests per second; the adapter enforces both
  (default 8 requests per second; with several worker processes the limit is
  shared between them).
* Amendments (`10-K/A`, `20-F/A`) are excluded unless `include_amendments=True`.
* **Fiscal year.** The default `jan_may_previous` convention assigns January–May period ends to the previous year. Set `--fiscal-year-rule period_end_year` (or the matching `EdgarSource` argument) to label every filing by its period-end year. The chosen rule is recorded in provenance and binds resume checkpoints.
* **Country.** The EDGAR state/country code of the business address is mapped
  to ISO 3166-1 alpha-2 with the SEC's code table (`A6` -> `CA`, `X0` -> `GB`,
  `2M` -> `DE`, U.S. states -> `US`; Puerto Rico, Guam and the U.S. Virgin
  Islands keep their own codes). The business address is where a company has
  its principal executive offices, which can differ from its country of
  incorporation or main operations.
* **Sections.** 10-K and 20-F documents are split into Items (`10-K:1A`,
  `20-F:4`, ...; see [Sections](sections.md)).
* **Exhibits** (`exhibits="auto"`, the default; CLI `--exhibits`):
    * a 10-K that incorporates content by reference to Exhibit 13 or to the
      annual report to shareholders gets that exhibit as section `10-K:EX-13`
      (selected together with Item 7 by section filters);
    * a 40-F, whose substance is in its exhibits, gets the exhibits described
      as Annual Information Form (`40-F:AIF`) and MD&A (`40-F:MDA`); if the
      filing index describes no exhibit that way, every `EX-99.*` exhibit
      whose description does not name financial statements, consents,
      certifications or press releases is added as `40-F:EX-99.x`.
    The exhibits are found through the filing index page
    (`<accession>-index.htm`), which costs one extra request per filing and is
    only fetched when needed. `exhibits="none"` disables this and
    `exhibits="all"` always adds Exhibit 13 and every `EX-99.*` exhibit.
* Industry: SIC code, mapped to a NACE Rev. 2 section for cross-country
  aggregation (`aidisclose.industry`; the mapping is approximate).

## filings.xbrl.org (EU/EEA and United Kingdom)

```python
src = aid.XbrlFilingsSource()
filings = src.list_filings(countries=["DK", "NL"], start_year=2021, end_year=2024)
```

* Lists ESEF and UKSEF inline-XBRL reports through the JSON:API
  interface of filings.xbrl.org (operated by XBRL International), following
  pagination links, and downloads the XHTML report.
* With `english_only=True` (default) reports whose file name indicates another
  language are dropped; reports without a language hint are screened after
  download and skipped if the text is not English.
* Annual status is not established by the index: it may contain interim reports. Review and select one annual report per firm-year; explicitly confirm in panel aggregation.
* One report per entity and period is kept (English first, then the most
  recently processed version).
* Reports are split into annual-report parts (`AR:management`, `AR:risk`,
  `AR:sustainability`, ...).
* Firm identifier: `lei:<LEI>`. Industry codes are not part of the index; join
  your own classification with `aidisclose panel --industry-map` or
  `firm_year_panel(results, industry_map=...)`.
* Coverage depends on what national storage mechanisms make available; the
  index itself notes that filings from some countries cannot be discovered
  reliably.

## Local folders (any jurisdiction)

```python
src = aid.LocalSource("reports/", metadata="reports/metadata.csv")
```

Supported formats: `.htm`, `.html`, `.xhtml`, `.xml`, `.txt`, `.pdf` (PDF needs
`pip install aidisclose[pdf]`; scanned PDFs without a text layer yield no text
and are flagged by `aidisclose qa`; OCR is not included and must be performed
before import). Mostly numeric HTML tables are removed to reduce extraction
noise, except tables whose cells mention AI-related terms, which are retained
for sentence-level screening. The metadata CSV has one row per file:

```
file,firm_id,firm_name,fiscal_year,country,industry_code,industry_scheme,form,period_end
```

`form=10-K` or `form=20-F` (or "10-K"/"20-F" in the file name) splits a file
into Items; all other files are split into annual-report parts. Every metadata
row must match one supported file; unmatched paths fail with a count and
examples instead of silently reducing coverage. If the same inferred firm ID
appears in different directories, provide explicit `firm_id` for those files
to establish whether they belong to the same firm. Fiscal years must be missing
or finite integer years in 1–9999; fractional or malformed values fail.
These checks do not establish whether the supplied identifiers are correct.

## Writing an adapter

An adapter subclasses `aidisclose.sources.Source` and implements two methods.
Use `HttpClient` for polite, cached downloads and `Source.make_document` for
text extraction, section splitting and language screening.

```python
from aidisclose.models import Filing
from aidisclose.sources import HttpClient, Source, register_source


class MyRegulatorSource(Source):
    name = "myregulator"

    def __init__(self, cache_dir=".aidisclose_cache/myregulator"):
        self.http = HttpClient("my-project contact@example.org", max_rps=2, cache_dir=cache_dir)

    def list_filings(self, start_year=None, end_year=None, **query):
        index = self.http.get_json("https://example.org/api/annual-reports", params={"from": start_year})
        return [
            Filing(source=self.name, filing_id=f"myreg:{r['id']}", firm_id=f"lei:{r['lei']}",
                   firm_name=r["name"], fiscal_year=r["year"], country="XX", url=r["url"], form="annual-report")
            for r in index["items"]
        ]

    def fetch_document(self, filing):
        data = self.http.get_bytes(filing.url)
        return self.make_document(filing, data, filename=filing.url)


register_source("myregulator", MyRegulatorSource)
```

Keep an `HttpClient` as an attribute of the source (as above): in
process-parallel runs AIDisclose finds it there and shares a request clock
among worker processes within the run. Separate runs are not coordinated.

To use the adapter from the command line, either load the module that
registers it with `--plugin`, or publish it as an entry point of your package:

```bash
aidisclose --plugin my_adapters run --source myregulator --source-option cache_dir=cache \
    --start 2022 --end 2024 --out results/myreg
```

```toml
[project.entry-points."aidisclose.sources"]
myregulator = "my_package.adapters:MyRegulatorSource"
```

`--source-option KEY=VALUE` is passed to the adapter's constructor and
`--query KEY=VALUE` to `list_filings()` (together with `--start`/`--end`).

Tests for adapters should replay recorded responses (see `tests/helpers.py`)
so that the test suite runs offline. `scripts/record_fixtures.py` records
current responses of the built-in sources, and `tests/test_live.py` checks the
adapters against the live services when `AIDISCLOSE_LIVE=1` is set.

### Local metadata identity and language

Metadata file entries must be unique. When files share a basename, provide their distinct relative paths; ambiguous basename fallback is rejected. A nonblank `language` in local metadata is treated as the explicit language code (case-insensitive), overriding text detection; without it, the detected language is used. This declaration determines language routing and must be verified by the researcher.

Period-end dates must be valid ISO calendar dates (`YYYY-MM-DD`) to derive a fiscal-year label; invalid dates yield a missing fiscal year.

## Source identity and dates

XBRL record IDs include the source record identifier (or a URL-based fallback) so language/version records remain distinct. Combine only results with verified compatible identifiers, and keep their samples and checkpoints associated with the matching dataset. When `languages` is specified, known filename hints outside the requested set are filtered before version selection; records without a hint are screened after download. `aidisclose list --all-languages` includes non-English hints, while `aidisclose run --languages ...` selects requested hints before deduplication. The user must supply and validate any non-English lexicon and classifier.

An invalid supplied SEC reporting period yields an unknown fiscal year. If the period is absent, a valid filing date can provide a filing-year-minus-one estimate; `extra.fiscal_year_basis` distinguishes this estimate from a reporting-period year. The estimate is not guaranteed to match the fiscal period. SEC request rates must be finite and in (0, 10] requests per second per run; separate runs are not coordinated.


XBRL records require a usable entity identifier. If entity metadata is absent,
only a structurally valid 20-character prefix in the source filing ID is used,
and `entity_identity_basis` marks it as unverified. Records with no usable
identifier cause an error instead of being assigned a fabricated shared firm.
Researchers should independently verify inferred and supplied identifiers.


Where an XBRL filing ID structurally encodes a 20-character entity prefix,
a conflicting included entity ID is rejected. Both forms are normalized to
uppercase, so affected lowercase IDs may change and should be regenerated.
This consistency check is not a checksum or external verification of identity.


The built-in LocalSource verifies the bytes actually read against the run-start
file digest; changed or initially unavailable files become failed documents.
A LocalSource subclass overriding `fetch_document` is not automatically
verified, so its own input consistency controls must be documented and tested.
This verification does not lock files or authenticate their authorship. HTTP
clients reject nonfinite, nonpositive or unusably small request rates.
