# Review XBRL report periods before annual aggregation

The filings.xbrl.org index includes both annual and interim reports, even when
an ESEF designation is present. A period end date or ESEF format alone does not
establish that a report covers a full year.

Users must review the source documents and select one
verified annual report per firm-year before panel aggregation. The CLI requires
`--confirm-xbrl-periods`; the Python API requires `xbrl_periods="reviewed"`.
Invalid period dates and multiple different XBRL period ends within one
firm-year are rejected even after confirmation. The confirmation records the
user's choice, not an automatic guarantee that the documents are annual; keep
a record of the period review and supporting source links.

## Real selected example

SOLAR A/S has both an interim report ending 30 June 2023 and an annual report
ending 31 December 2023. The reports were downloaded from filings.xbrl.org on
27 October 2026 and processed independently; this convenience example is not a
representative sample or a classifier accuracy study.

- Interim XHTML: https://filings.xbrl.org/21380031XTLI9X5MTY92/2023-06-30/ESEF/DK/0/SOLA-2023-06-30-en/reports/SOLA-2023-06-30-en.xhtml
  SHA-256 `fa92a27aa07437dc4cd32b1c6cc1fe7e284cf09495b7ec065e8d650d47a3b6b1`.
- Annual XHTML: https://filings.xbrl.org/21380031XTLI9X5MTY92/2023-12-31/ESEF/DK/0/SOLA-2023-12-31-en/reports/SOLA-2023-12-31-en.xhtml
  SHA-256 `bf17bb233b2d68166e41cecb1923ce45eabe868620966cb1afc9162b7e6ccdf3`.
- The publisher identifies the first as its six-month interim report:
  https://attachment.news.eu.nasdaq.com/a04cd757a60628f479afb7075cdc00947
  and the second as the annual report:
  https://view.news.eu.nasdaq.com/view?id=b8e0fb79b852c594bc48f810f9a5ee414&lang=en&src=micro

After independently downloading the source files, reproduce the guard:

```bash
python examples/06_xbrl_period_review.py --annual /path/to/annual.xhtml --interim /path/to/interim.xhtml
```

This selected example illustrates the period-review workflow; it is not a
prevalence or accuracy estimate. Raw filings and text-bearing results are not
redistributed; consult the relevant source terms.
