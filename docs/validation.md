# Validation

Machine labels are measurements and must be validated on the corpus at hand.
AIDisclose supports four checks: precision-oriented annotation of retrieved
sentences, inter-annotator agreement, a recall audit of sentences the lexicon
did not retrieve, and comparison with external statistics.

## 1. Annotating retrieved sentences

```bash
aidisclose sample --results results/ --n 400 --strata state --min-per-stratum 40 \
    --out to_annotate.xlsx --label-studio to_annotate_ls.json
```

* `to_annotate.xlsx` contains the sentences with context but **no
  predictions**, so annotators are not anchored by the classifier. The
  workbook has drop-down lists for `gold_state` and `gold_reason` and a
  codebook sheet. A `.csv` name writes UTF-8 with a byte-order mark, which
  Excel opens correctly.
* `to_annotate_key.csv` keeps the predictions and a `sampling_weight` per row
  (stratum size divided by the number of sampled rows), because rare states
  are over-sampled.
* `--label-studio` also writes Label Studio tasks and a labelling
  configuration (`*_config.xml`); `aidisclose import-label-studio --export
  export.json --out annotated.csv` converts the Label Studio export back, one
  row per annotator.

Annotate `gold_state` (and optionally `gold_reason`) following the
[codebook](codebook.md), then evaluate:

```bash
aidisclose evaluate --gold annotated.xlsx --results results/ --key to_annotate_key.csv --weighted --out report.md
```

The report contains per-state precision, recall and F1, macro-F1, accuracy,
Cohen's kappa between gold labels and predictions, two binary views that
correspond to the panel indicators (SUBSTANTIVE vs rest; claim vs no claim)
and bootstrap confidence intervals for every one of these numbers. By default,
the CLI resamples whole filings (`--cluster-by filing_id`); choose
`--cluster-by firm_id` for firm clusters or `--cluster-by none` for row
resampling. With `--key ... --weighted`, estimates use sampling weights. The
cluster interval is a percentile bootstrap over observed clusters and should be
read as an approximate uncertainty measure when clusters are few or the sample
design is complex.
`evaluate` also checks that each annotated sentence has the same text as the
matched prediction. Classifier-error rows are excluded and counted in the CLI
warning.

For supervised models, `aidisclose train --cv 5 --cv-group-by firm_id` keeps
each firm's classifier-eligible sentences within one fold and uses a cluster bootstrap for its
intervals. `filing_id` is also available. `--cv-group-by sentence` reproduces
random sentence folds and can leak report-specific language across folds.
Sample-key training requires `--results` from the original extraction; baseline-only
and filtered rows are excluded, and the CV report covers the eligible subset.
Evaluate the full pipeline independently to assess overall performance.

Note that "recall" in this report is recall *among retrieved sentences*;
sentences the lexicon never retrieved are covered by the recall audit below.

## 2. Inter-annotator agreement

Have two or more people label the same rows independently, then:

```bash
aidisclose agreement --annotations ann_alice.xlsx ann_bob.xlsx --out agreement.md
aidisclose agreement --label-studio export.json          # several annotators in one export
```

The report gives percent agreement, Cohen's kappa for each pair (with the
confusion matrix), Fleiss' kappa and Krippendorff's alpha (nominal; missing
ratings allowed), and lists the disagreements for adjudication. The
implementations reproduce the textbook examples for Fleiss' kappa (0.210) and
Krippendorff's alpha (0.743) in the test suite. When both raters use only the
same single label, Cohen's kappa is undefined (NaN), even if percent agreement
is 100%; the same rule applies to weighted evaluation.

## 3. Recall audit

Retrieval by lexicon misses AI claims that use none of its terms ("our
routing algorithms", a product name). To estimate how many, keep a recall pool
during the run:

```bash
aidisclose run ... --recall-random 5 --recall-probe-cap 10 --out results/
aidisclose sample-recall --results results/ --n 300 --share-probe 0.5 --out recall_sample.xlsx
# annotate gold_state in recall_sample.xlsx
aidisclose evaluate-recall --gold recall_sample.xlsx --key recall_sample_key.csv \
    --main-gold annotated.xlsx --main-key to_annotate_key.csv --out recall.md
```

* Per document the pipeline keeps up to `--recall-probe-cap` sentences that
  contain an AI-adjacent probe term ("algorithm", "automation", "predictive",
  "robot", bare product names, ...; lexicon `recall-probe`) but no
  measurement term (including baseline-only hits), and `--recall-random` of the other
  non-retrieved sentences, with their within-document inclusion
  probabilities. The draws are reproducible (seeded by filing id).
* `sample-recall` draws an annotation sample from the pool, over-sampling
  the probe stratum, and records each row's overall inclusion probability.
* `evaluate-recall` computes Horvitz-Thompson estimates of the number of
  missed SUBSTANTIVE sentences and missed claims (SUBSTANTIVE or WEAK), and,
  combined with the number of retrieved true SUBSTANTIVE sentences estimated
  from the main annotated sample, the lexicon's **recall**
  `found / (found + missed)`, with stratified bootstrap intervals over both
  samples. Without `--main-gold`, `--results` uses predicted SUBSTANTIVE and
  WEAK counts for the respective recall measures; these counts are unvalidated.
* The estimates refer to the processed sections of the processed (English)
  documents. Missed claims are rare, so the random stratum needs a large
  sample for a tight interval; the probe stratum shows *where* misses occur
  and which terms to consider for the next lexicon version.

## 4. External benchmarks

Eurostat's survey on ICT usage in enterprises reports the share of
enterprises using at least one AI technology (indicator `E_AI_TANY`, unit
`PC_ENT`):

* `isoc_eb_ai`: by country and size class; listed firms are large, so
  compare with `GE250`;
* `isoc_eb_ain2`: by country and NACE section, for enterprises with 10 or more
  persons employed only, and without the financial sector (K).

```bash
aidisclose eurostat --panel country_year.csv --eurostat-file isoc_eb_ai.json --out eurostat.csv
aidisclose panel --results results/ --level country-industry-year --out cny.csv
aidisclose eurostat --dataset isoc_eb_ain2 --panel cny.csv --eurostat-file isoc_eb_ain2.json --out eurostat_ind.csv
```

```python
from aidisclose.validate import compare_with_external, eurostat_ai_share, eurostat_ai_share_by_industry

ext = eurostat_ai_share(size_emp="GE250")        # downloads from the Eurostat API, or path="isoc_eb_ai.json"
merged, summary = compare_with_external(country_year_panel, ext, min_firms=10)
```

Any other benchmark (for example a national business survey of AI use) can be
compared from a CSV file; map its industry codes to your panel's
classification first:

```bash
aidisclose benchmark --panel country_year.csv --file survey.csv --value ai_use_rate \
    --rename geo=country ref_year=year --scale percent --out survey_comparison.csv
```

The summary reports, for the keyword, claim and substantive prevalence, the
Spearman and Pearson correlations with the benchmark, a permutation p-value
and the mean absolute difference in percentage points. The comparison is
indicative: survey respondents and listed companies differ in size, sector mix
and the definition of AI use, and English-language reports are not a random
sample of a country's listed firms. Country-year panels use a block permutation
that reassigns complete country trajectories when the panel is balanced; the
p-value is omitted for unbalanced panels. Generic benchmarks without country
blocks report a row permutation and should be interpreted cautiously.

## Quality checks

```bash
aidisclose qa --results results/ --out qa.md
```

`qa` lists failed and empty documents, 10-K/20-F filings whose Items were not
found, over-long sentences (tables read as text), sentences repeated across
many filings, the most frequent false-positive terms, classifier errors,
skipped languages, missing fiscal years, several reports per firm-year and
coverage gaps. Run it before computing indicators; `--strict` exits with an
error if there are warnings (for scripted pipelines).

## Sampling safeguards

Annotation allocation never exceeds the requested size and gives each nonempty stratum positive inclusion probability; a sample smaller than the number of strata is rejected. Recall sampling rejects invalid inclusion probabilities and allocations that omit a nonempty pool stratum. Recall estimation requires complete valid annotations and finite positive weights. The procedure cannot recover strata excluded at the initial pool stage: set both recall caps above zero for each document and verify coverage before interpreting population estimates. Its row-stratified bootstrap is an approximation and does not fully reproduce the two-stage document sampling design; independent design-based variance validation remains necessary.

## Integrity and coverage safeguards

The weighted CLI requires the complete matching sample key, unique sentence identifiers, unchanged text, valid gold and predicted labels, and finite positive weights. It fails rather than silently substituting an unweighted estimate or discarding missing labels. Macro-F1 averages F1 across classes appearing in the gold labels or predictions: a class present but never predicted receives F1 = 0; a class absent from both is excluded.

`sample-recall` checks every processed document's per-stratum population, draw size and sampling probability. It refuses a nonempty stratum with zero inclusion probability or a missing or inconsistent pool, and marks the resulting key. `evaluate-recall` requires that key and matching complete annotations; old keys cannot establish first-stage coverage and must be regenerated. This check does not prove that any annotator is correct or that sampling models all disclosure documents. The current row-stratified bootstrap provides approximate intervals, not independently validated design-based intervals for the two-stage procedure. User-supplied API data must independently establish coverage.

A country-block permutation preserves original row positions for interleaved blocks. Non-exact Monte Carlo permutation p-values use the plus-one correction.

## Sample identity and complete votes

Sample and recall key files carry a dataset digest. Weighted evaluation and recall estimation require keys from the same exact loaded result set, complete labels and positive finite weights; samples that cannot be verified must be regenerated. Gold input must not contain prediction columns. First-stage recall sampling fails if classifier errors are present. The recall population is defined by absence of a main-lexicon raw hit, so a sentence matched only by the keyword baseline remains eligible. Recall bootstrap intervals still approximate the two-stage design and have not been independently validated as design-based intervals.

## Completeness, intervals and missing outcomes

CLI-generated sample keys have a row count and SHA-256 completeness seal over selected key fields, so simultaneously deleting the same rows from the key and annotation file is detected. Regenerate keys that cannot be verified against the associated result dataset. This guards against accidental changes, not intentional alteration.

A recall bootstrap interval is not produced when a recall or main sampling stratum has fewer than two annotated rows. Other recall intervals remain approximate and are not independently validated for the full two-stage design. Group claim and substantive shares are missing when a positive-weight firm's value is unknown. Observed-only shares and worst-case lower and upper bounds are reported separately; observed-only values are conditional on being observed and need not be unbiased.

## Benchmark keys

Comparison requires complete, unique keys on both the disclosure panel and the benchmark. Aggregate duplicate observations or choose more specific matching keys before comparison; missing keys cannot be matched.

Percent-marked benchmark values are interpreted as percentages, including values below 1%. Values at or below 1 without a percent sign remain ambiguous; set the percent option explicitly when they represent percentages. Shares outside [0, 1] after conversion are rejected.

## Classification coverage and agreement

Weighted population evaluation requires complete classification throughout the result set; resolve classifier errors and generate a new sample first. When a row-bootstrap stratum has a single annotation, the point estimate remains available but its interval is not estimated. Cluster bootstrap remains a separate approximation. Agreement reports require unique nonblank item identifiers per annotator, valid state labels and nonblank annotator identifiers; missing ratings are excluded, not treated as a category. Local Eurostat industry files must include size_emp and are filtered to GE10 and any requested countries.

Sampling stratum identifiers are encoded unambiguously; if an older sample used multiple stratification fields with delimiter-like text or mixed types, regenerate it to avoid conflated strata in row-bootstrap intervals.

## LLM reason validation

LLM responses must contain a recognized reason compatible with the state. Invalid or missing reasons are retried and, if unresolved, reported as classifier errors. Invalid cached responses are discarded. Supervised or transformer backends may still return UNSPECIFIED when they provide only a state.

JSON-stat datasets with no reported observations retain their dimension columns and return a correctly headed empty comparison input.


External comparison inputs must contain finite numeric values or missing
observations. Benchmark CSV values may use empty, NA, N/A, NULL, NaN or `.`
for missing values; other malformed or nonfinite values raise an error rather
than silently reducing the comparison sample. Direct permutation calls require
nonmissing block identifiers and a supported correlation method. These checks
do not establish the suitability or independence of an external benchmark.


Local Eurostat JSON-stat files must retain the selected size, indicator and
unit dimensions; records are filtered to the requested codes and percentage
values must lie in [0, 100]. Files without those dimensions cannot establish
the intended comparison population or measure. For predicted retrieved totals,
claim recall uses both SUBSTANTIVE and WEAK predictions.


Country-level Eurostat comparisons explicitly select `C10-S951_X_K` by
default and retain `nace_r2` in the output. Supply `--nace-aggregate` (or
`nace_r2` in Python) to choose another scope; local files lacking that dimension
cannot verify the selection. This aggregate excludes some activities and must
not be described as all firms. Saved recall reports label counts derived from
model predictions as unvalidated.


For `isoc_eb_ain2`, the CLI uses GE10; an explicitly requested other size or
country-level `--nace-aggregate` is rejected rather than ignored. Country-level
`isoc_eb_ai` uses GE250 and `C10-S951_X_K` by default and accepts explicit
alternatives. Inspect the selected scope when interpreting comparisons.


Sample keys use schema 2 and seal available annotation context (company,
year, section and adjacent sentences) as well as the target sentence and
sampling fields. Evaluation and training compare context fields present in
both the annotation and key. Missing annotation fields trigger a warning;
legacy keys remain readable but did not seal context and also trigger a warning.
Label Studio imports preserve task context when present. These hashes detect
accidental changes, not deliberate tampering or annotator correctness.


Sample, recall-sample and report commands reject output paths that alias each
other or protected result files; report Markdown and JSON paths must differ.
Training also checks the model and out-of-fold paths against annotations and
keys. Agreement rejects using the same file as two independent annotators.
File-based agreement labels are consistent across summaries, pairs and
disagreement records, including when input filenames have the same stem.

Weighted CLI evaluation requires `sentence_id` among `--on` join columns so sample weights remain attached to their original predictions.
