# Indicators

```bash
aidisclose panel --results results/ --level firm-year --out firm_year.csv
aidisclose panel --results results/ --level country-year --min-firms 10 --out country_year.csv
aidisclose panel --results results/ --level industry-year --firm-attributes employees.csv --weight employees \
    --out industry_year_weighted.csv
```

```python
panel = aid.firm_year_panel(results, industry_map=None, firm_attributes=None)
groups = aid.group_panel(panel, by=["country", "fiscal_year"], weight=None, min_firms=1)
```

When a firm has several processed reports for the same fiscal year (e.g. a
10-K and a 10-K/A), the longest report is kept. A processed report with an
unknown fiscal year prevents panel creation by default. Correct the metadata
and rerun, or explicitly choose `unknown_years="exclude"` in Python or
`--exclude-unknown-years` in the CLI. Exclusion issues a warning with the
number of reports removed; excluded reports are absent from all panel totals.

## Firm-year

| Column | Definition |
|---|---|
| `keyword_any` | the report contains "artificial intelligence", "machine learning", "AI" or "ML" (plain keyword baseline, no disambiguation) |
| `claim_any` | at least one WEAK or SUBSTANTIVE sentence |
| `substantive_any` | at least one SUBSTANTIVE sentence |
| `n_substantive`, `n_weak`, `n_no_claim`, `n_false_positive` | sentence counts by state |
| `keyword_per_10k_words`, `substantive_per_10k_words` | intensities scaled by report length |
| `substantive_share` | SUBSTANTIVE sentences / sentences with any lexicon hit |
| `n_substantive_<category>`, `substantive_<category>_any` | SUBSTANTIVE sentences with a term of that lexicon category: `general`, `ml`, `genai`, `nlp`, `cv`, `automation` (a sentence can count for several) |
| `n_substantive_deployment`, `_product`, `_investment` | SUBSTANTIVE sentences by reason (rule and LLM backends) |
| `n_weak_forward_looking`, `_hedged`, `_risk_boilerplate`, `_vague` | WEAK sentences by reason |
| `n_substantive_new`, `substantive_novelty` | SUBSTANTIVE sentences that do not repeat a lexicon-hit sentence of the previous year's report, and their share; see below |
| `first_observed_year`, `first_substantive_year`, `substantive_onset` | first year in the panel, first SUBSTANTIVE year, and an onset flag; see below |
| `evidence` | up to three SUBSTANTIVE sentence ids, highest confidence first (`Results.evidence()` returns all) |
| `n_classifier_errors` | sentences the backend could not label |
| `classification_complete`, `n_classified_sentences` | whether all retrieved candidate sentences received usable labels, and the number that did |

When any classifier error occurs, `claim_any` and `substantive_any` remain 1
if a known positive label exists; otherwise they are missing because absence
has not been established. Intensity and share measures that require a complete
classification are missing. Group panels report `n_firms_complete`,
`n_firms_incomplete`, and the observed denominators `n_firms_claim_observed`
and `n_firms_substantive_observed`. A binary indicator with a known positive
remains usable even in an otherwise incomplete filing; unknown zeros are
omitted from observed-only group means. Full-group claim and substantive shares
are missing when any positive-weight value is unknown. The keyword–substantive
gap is also missing in that case; otherwise its two prevalences have a matched
positive-weight denominator. Counts of known positive sentences are
lower bounds when classification is incomplete.

### Novelty

Annual reports repeat text from year to year. A SUBSTANTIVE sentence counts as
*new* if no lexicon-hit sentence in the firm's report for the previous fiscal
year has a token-set Jaccard similarity of at least `repeat_threshold`
(default 0.8; digits are ignored, so "cut fuel use by 7% in 2023" restated as
"by 9% in 2024" is a repetition). Novelty is empty when the previous fiscal
year is not in the panel or there is no SUBSTANTIVE sentence. It separates
firms that report new AI activity from firms that carry forward the same
paragraph.

### Onset

`substantive_onset` is 1 in a firm's first observed SUBSTANTIVE year in the panel if
the previous fiscal year was observed (necessarily without a SUBSTANTIVE
claim), empty in that year if the previous year was not observed (the onset
is censored: the firm may have made claims before the panel starts or in the
missing year), and 0 in all other years. Count onsets only where they are
observed.

## Group-year

`group_panel` returns, per group: `n_firms`, `share_keyword`, `share_claim`,
`share_substantive`, `share_keyword_substantive_observed`,
`share_substantive_<category>`, `mean_*` intensities,
`mean_substantive_novelty`, `share_onset` and `gap_keyword_substantive_pp`.
The gap is keyword prevalence minus substantive prevalence in percentage
points, and is missing when a positive-weight substantive value is unknown.
Observed-only shares remain available and describe only observed firms.

Means are over firms (each firm-year has weight 1) or weighted by a numeric
column: `--weight employees` after merging `--firm-attributes` (a CSV with
`firm_id`, optionally `fiscal_year`, and any other columns; names that clash
with panel columns get an `attr_` prefix). Firms with a missing or
non-positive weight are left out of weighted means; `n_firms_weighted` says how
many remain. Use `--min-firms` to suppress groups with few firms.

## Industry

`nace_section` comes from the filing (NACE codes in local metadata), from the
SEC's SIC code through an approximate SIC -> NACE Rev. 2 section mapping, or
from your own table: `--industry-map map.csv` with columns `industry_code`
and `group` (or `nace_section`) overrides it.

## Incomplete group-level means

`mean_keyword_per_10k_words`, `mean_substantive_per_10k_words` and `mean_substantive_share` describe the full positive-weight group only when every contributing firm has a defined value. Otherwise these means are missing. Separate `*_observed` means and `n_firms_*_observed` counts refer only to positive-weight firms with defined values. Observed-only values can be selected or biased. In particular, `substantive_share` is undefined for a report without measurement-lexicon hits; a group containing such a positive-weight report therefore has no full-group mean for this quantity. A group eliminated by `min_firms` or non-positive weights returns an empty table with headers.
