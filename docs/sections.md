# Sections

Every document is split into labelled sections before sentences are
extracted. Section labels appear in `sentences.csv`, in sentence identifiers
(`<filing_id>#<section>#<index>`) and in the rule backend, which treats
risk-factor sections differently (codebook rules 4 and 10).

## Item-structured SEC forms

| Form | Labels | Risk section |
|---|---|---|
| 10-K | `10-K:cover`, `10-K:1`, `10-K:1A`, `10-K:1B`, `10-K:1C`, `10-K:2` ... `10-K:16` | `10-K:1A` |
| 20-F | `20-F:cover`, `20-F:1` ... `20-F:4`, `20-F:4A`, `20-F:5` ... `20-F:16`, `20-F:16A` ... `20-F:16K`, `20-F:17` ... `20-F:19` | `20-F:3` (incl. 3.D Risk Factors) |
| exhibits | `10-K:EX-13`, `40-F:AIF`, `40-F:MDA`, `40-F:EX-99.x`, `40-F:main` | |

Item headings appear twice, in the table of contents and at the start of each
Item, and cross-references ("see Item 7") can look like headings. The splitter
finds every heading line (also combined headings such as "Items 1 and 2." and
run-in headings such as "Item 1. Business. The Company ..."), starts the body
at the first-Item heading that opens the longest span (the table-of-contents
entry opens a short one) and keeps the longest sequence of headings with
increasing Item numbers, which discards cross-references. If fewer than three
Items are found the document is kept whole (and flagged by `aidisclose qa`).

## Free-form annual reports

ESEF/UKSEF reports, PDFs and other annual reports are split into parts by
their headings:

| Label | Typical headings |
|---|---|
| `AR:front` | everything before the first recognised heading (cover, contents) |
| `AR:management` | strategic report, management report/review, directors' report, letter from the CEO/chair, business review, financial review, highlights |
| `AR:risk` | principal risks (and uncertainties), risk management, risk factors |
| `AR:sustainability` | sustainability statement/report, non-financial statement, ESG report, CSRD/ESRS |
| `AR:governance` | corporate governance, remuneration report |
| `AR:financial_statements` | consolidated financial statements, notes to the financial statements |
| `AR:auditor` | independent auditor's report |

A heading counts only if at least 30 words follow it before the next heading,
which ignores tables of contents and running headers. Blocks of the same part
are concatenated. If fewer than two parts are found the document is kept whole
(section `full`). Heading conventions differ between countries and firms, so
check the section mix of a new corpus (`aidisclose qa`, or `sentences.csv`)
before relying on section filters.

## Section filters and presets

```python
aid.Pipeline(sections=["10-K:1", "10-K:1A", "10-K:7"])
aid.Pipeline(sections="core")                       # preset
aid.Pipeline(sections=["core", "AR:sustainability"]) # preset plus extra labels
```

```bash
aidisclose run ... --sections core
```

A filter restricts only the document types it names: `["10-K:1A"]` limits
10-K filings to Item 1A and leaves 20-F filings and annual reports whole.
Unsplit documents (`full`) and the main text of filings whose Items could not
be found (`<form>:main`) are always processed. Exhibit 13 (`10-K:EX-13`) is
selected whenever `10-K:7` is, because it carries the MD&A of firms that
incorporate it by reference.

| Preset | Labels |
|---|---|
| `core` | 10-K Items 1, 1A, 7; 20-F Items 3, 4, 5; 40-F AIF and MD&A; `AR:front`, `AR:management`, `AR:risk` |
| `narrative` | `core` plus 10-K Items 1C, 7A, 9A, 10; 20-F Items 6, 11, 15, 16G, 16K; `AR:sustainability`, `AR:governance` |

Report the section filter you used: indicators computed on different sections
are not comparable.
