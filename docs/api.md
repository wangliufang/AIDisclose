# Python API

Generated from the docstrings by `scripts/gen_docs.py`.

## Pipeline and results

`aidisclose.pipeline`

### `Pipeline`

```python
class Pipeline(backend: 'str | Classifier' = 'rules', lexicon: 'str | Path | Lexicon' = 'default', tiers: 'Iterable[str] | None' = ('core',), baseline_lexicon: 'str | Path | Lexicon' = 'keyword-baseline-4', sections: 'Iterable[str] | str | None' = None, english_only: 'bool' = True, backend_options: 'dict[str, Any] | None' = None, max_sentence_chars: 'int' = 1200, languages: 'Iterable[str] | None' = None, lexicons: 'Mapping[str, str | Path | Lexicon] | None' = None, baseline_lexicons: 'Mapping[str, str | Path | Lexicon] | None' = None, recall_random: 'int' = 0, recall_probe_cap: 'int' = 0, recall_seed: 'int' = 20240601, probe_lexicon: 'str | Path | Lexicon' = 'recall-probe') -> 'None'
```

Measure AI adoption claims in disclosure documents.

**Parameters**

backend:
    `"rules"` (default), `"supervised"`, `"llm"` or a
    :class:`~aidisclose.classify.base.Classifier` instance.
lexicon:
    Built-in lexicon name, path to a YAML lexicon or a :class:`Lexicon`
    (used for English, and for every language when `languages` is
    `None` and `english_only` is `False`).
tiers:
    Lexicon tiers to use (`("core",)` by default, `None` for all).
baseline_lexicon:
    Lexicon used to compute the plain keyword baseline (English).
sections:
    Section labels to process (e.g. `["10-K:1", "10-K:1A", "10-K:7"]`)
    or a preset name (`"core"`, `"narrative"`). A filter restricts
    only the document types it names; documents without sections are
    always processed in full (see :class:`~aidisclose.text.sections.SectionFilter`).
english_only:
    Skip documents whose detected language is not English (ignored when
    `languages` is given).
languages:
    Languages to process, e.g. `("en", "de")`. Every language other than
    English needs an entry in `lexicons`.
lexicons:
    `{language: lexicon}` for languages other than English.
baseline_lexicons:
    `{language: lexicon}` keyword baselines for other languages (default:
    the language's measurement lexicon).
recall_random, recall_probe_cap, recall_seed, probe_lexicon:
    Recall pool: per document, keep up to `recall_probe_cap` sentences
    that match the probe lexicon but not the measurement lexicon, and
    `recall_random` of the remaining non-retrieved sentences, drawn with
    a seed derived from `recall_seed` and the filing id.
backend_options:
    Keyword arguments passed to the backend constructor.

**`Pipeline.analyse(self, doc: 'Document') -> 'DocumentOutcome'`**

Classify one document, including its recall-pool sample.

**`Pipeline.candidates_from_results(self, results: 'Results') -> 'list[Candidate]'`**

Rebuild the classified candidates of an earlier run (rows with valid terms).

**`Pipeline.extract_candidates(self, doc: 'Document') -> 'tuple[list[Candidate], dict[str, int]]'`**

Split a document into sentences and keep those with lexicon hits.

**`Pipeline.fingerprint(self, source_config: 'Mapping[str, Any] | None' = None) -> 'str'`**

Digest of the configuration that determines the output (for checkpoints).

**`Pipeline.process_document(self, doc: 'Document') -> 'tuple[list[dict[str, Any]], dict[str, Any]]'`**

Classify one document; returns `(sentence rows, document row)`.

**`Pipeline.process_filing(self, filing: 'Filing', source, *, verify_input: 'bool' = False, expected_sha256: 'str | None' = None) -> 'DocumentOutcome'`**

Fetch and analyse one filing; failures are recorded, not raised.

**`Pipeline.relabel(self, results: 'Results') -> 'Results'`**

Re-classify the candidate sentences of an earlier run with this pipeline's backend.

Documents are not downloaded or segmented again: the sentences, their
lexicon terms and the document rows are reused, and only the state,
reason, confidence, backend and details columns (and the per-document
counts) change. Use it to compare backends on identical candidates.

**`Pipeline.run(self, filings: 'Iterable[Filing]', source, progress: 'bool' = False, workers: 'int' = 1, executor: 'str' = 'thread', checkpoint_dir: 'str | Path | None' = None, resume: 'bool' = False, retry_errors: 'bool' = True) -> 'Results'`**

Fetch and process filings from a :class:`~aidisclose.sources.base.Source`.

**Parameters**

workers, executor:
    Number of concurrent workers and `"thread"` or `"process"`.
checkpoint_dir:
    Directory for one checkpoint file per filing. Without `resume`
    any existing checkpoints there are discarded first.
resume:
    Reuse checkpoints written by an earlier run with the same
    configuration; filings that failed are retried unless
    `retry_errors` is false.

### `Results`

```python
class Results(sentences: 'pd.DataFrame', documents: 'pd.DataFrame', provenance: 'dict[str, Any]' = <factory>, recall_pool: 'pd.DataFrame' = <factory>) -> None
```

Sentence- and document-level output of a pipeline run.

`recall_pool` holds the sentences kept for the recall audit (empty unless
the pipeline was configured to keep them).

**`Results.dataset_digest(self) -> 'str'`**

Bind samples to the exact loaded sentence, document and recall tables.

**`Results.evidence(self, firm_id: 'str | None' = None, fiscal_year: 'int | None' = None, state: 'str' = 'SUBSTANTIVE') -> 'pd.DataFrame'`**

Return the sentences behind an indicator value (the evidence trail).

### `DocumentOutcome`

```python
class DocumentOutcome(rows: 'list[dict[str, Any]]', doc_row: 'dict[str, Any]', recall_rows: 'list[dict[str, Any]]' = <factory>) -> None
```

Everything the pipeline produces for one filing.

## Indicators

`aidisclose.aggregate`

### `firm_year_panel`

```python
function firm_year_panel(results, industry_map: 'Mapping[str, str] | Callable | None' = None, n_evidence: 'int' = 3, firm_attributes: 'pd.DataFrame | None' = None, repeat_threshold: 'float' = 0.8, unknown_years: 'str' = 'error', xbrl_periods: 'str' = 'error') -> 'pd.DataFrame'
```

Build a firm-year panel from :class:`~aidisclose.pipeline.Results`.

When a firm has several processed reports for the same fiscal year (e.g. a
10-K and a 10-K/A), the longest report is kept.

`firm_attributes` is merged on `firm_id` (and `fiscal_year` if it has
that column), e.g. to add employment for weighted group means.
Unknown fiscal years raise an error unless `unknown_years="exclude"`;
exclusion emits a warning with the number of reports removed.
XBRL indexes can contain interim reports; users must verify that every
included XBRL report is annual before passing `xbrl_periods="reviewed"`.
Multiple XBRL period ends for the same firm-year are rejected even after
confirmation; select one verified report per firm-year before processing.

### `group_panel`

```python
function group_panel(firm_panel: 'pd.DataFrame', by: 'Iterable[str]' = ('country', 'fiscal_year'), weight: 'str | None' = None, min_firms: 'int' = 1) -> 'pd.DataFrame'
```

Aggregate a firm-year panel to groups such as country-year or industry-year.

`weight` names a numeric column (e.g. employees merged through
`firm_attributes`); firms with a missing or non-positive weight are left
out of weighted means, and `n_firms_weighted` reports how many remain.

### `state_mix`

```python
function state_mix(results, by: 'Iterable[str]' = ('fiscal_year',)) -> 'pd.DataFrame'
```

Share of keyword sentences by state and reason (what keyword counts are made of).

### `read_industry_map`

```python
function read_industry_map(path) -> 'dict[str, str]'
```

Read a CSV with columns `industry_code` and `group` (or `nace_section`).

## Sources

`aidisclose.sources`

### `Source`

```python
class Source()
```

A provider of disclosure documents.

**`Source.config(self) -> 'dict[str, Any]'`**

Return the non-secret source settings used for provenance.

**`Source.fetch_document(self, filing: 'Filing') -> 'Document'`**

Download (or read) a filing and return its extracted text.

**`Source.list_filings(self, **query: 'Any') -> 'list[Filing]'`**

Return metadata for the filings matching *query*.

**`Source.make_document(filing: 'Filing', data: 'bytes', filename: 'str' = '', sections_fn='auto') -> 'Document'`**

Extract text from *data* and split it into sections.

`sections_fn="auto"` (default) uses the Item schema of the filing's
form (10-K, 20-F) or, for other documents, the annual-report part
splitter; `None` keeps the document whole; a callable
`text -> {label: text}` is used as given.

### `HttpClient`

```python
class HttpClient(user_agent: 'str', max_rps: 'float' = 5.0, cache_dir: 'str | Path | None' = None, timeout: 'float' = 60.0, max_retries: 'int' = 4, session: 'requests.Session | None' = None) -> 'None'
```

HTTP GET with rate limiting, retries and an optional on-disk cache.

**Parameters**

user_agent:
    Value of the `User-Agent` header. Some regulators (e.g. the SEC)
    require a declared agent with contact details.
max_rps:
    Maximum requests per second sent to the network.
cache_dir:
    Directory where raw responses are stored. Cached files are reused on
    later runs, which makes a study reproducible even if a source changes.

### `EdgarSource`

```python
class EdgarSource(user_agent: 'str', cache_dir: 'str | Path | None' = '.aidisclose_cache/edgar', max_rps: 'float' = 8.0, session=None, exhibits: 'str' = 'auto', fiscal_year_rule: 'str' = 'jan_may_previous')
```

List and download annual reports (10-K, 20-F, 40-F) from SEC EDGAR.

**Parameters**

user_agent:
    Required by the SEC, e.g. `"Jane Doe jane.doe@university.edu"`.
cache_dir:
    Directory for cached API responses and documents.
max_rps:
    Request rate limit (must stay below the SEC maximum of 10).
exhibits:
    `"auto"` (default; see module docstring), `"none"` or `"all"`
    (always add EX-13 to 10-K filings and all EX-99.* to 40-F filings).

**`EdgarSource.config(self) -> 'dict[str, Any]'`**

Return the non-secret source settings used for provenance.

**`EdgarSource.fetch_document(self, filing: 'Filing') -> 'Document'`**

Download (or read) a filing and return its extracted text.

**`EdgarSource.list_filings(self, ciks: 'Iterable[int | str] | None' = None, tickers: 'Iterable[str] | None' = None, forms: 'Iterable[str]' = ('10-K',), start_year: 'int | None' = None, end_year: 'int | None' = None, include_amendments: 'bool' = False) -> 'list[Filing]'`**

List annual reports for the given companies.

`forms` may include `"10-K"`, `"20-F"` and `"40-F"`.
`start_year`/`end_year` filter on the fiscal year (derived from the
reporting period; periods ending January-May count as the previous
fiscal year). When the period is absent, a valid filing date may be
used for an explicitly marked filing-year-minus-one estimate. An
invalid supplied period remains unknown. Amendments (`.../A`) are
excluded unless requested.

**`EdgarSource.lookup_ciks(self, tickers: 'Iterable[str]') -> 'dict[str, int]'`**

Map ticker symbols to CIK numbers using the SEC ticker file.

### `XbrlFilingsSource`

```python
class XbrlFilingsSource(user_agent: 'str' = 'aidisclose/0.5.3 (research software; set a contact address)', cache_dir: 'str | Path | None' = '.aidisclose_cache/xbrl', max_rps: 'float' = 2.0, session=None) -> 'None'
```

List and download ESEF/UKSEF reports from filings.xbrl.org.

**`XbrlFilingsSource.fetch_document(self, filing: 'Filing') -> 'Document'`**

Download (or read) a filing and return its extracted text.

**`XbrlFilingsSource.list_filings(self, countries: 'Iterable[str] | None' = None, start_year: 'int | None' = None, end_year: 'int | None' = None, leis: 'Iterable[str] | None' = None, english_only: 'bool' = True, page_size: 'int' = 200, max_pages: 'int | None' = None, keep_all_versions: 'bool' = False, languages: 'Iterable[str] | None' = None) -> 'list[Filing]'`**

List reports by country and fiscal year, or by LEI.

With `english_only` (default) reports whose file name indicates
another language are dropped; reports without a language hint are kept
and screened after download. `languages`, when supplied, selects
acceptable hints and takes precedence over `english_only`; its order
determines language preference during deduplication. Unless
`keep_all_versions` is set, one report per entity and valid period
is kept. Unknown periods are not merged across different records.
This listing does not verify whether a report is annual or interim.

### `LocalSource`

```python
class LocalSource(directory: 'str | Path', metadata: 'str | Path | None' = None, recursive: 'bool' = True)
```

Read reports from a local folder.

10-K and 20-F documents (`form` column, or "10-K"/"20-F" in the file
name) are split into Items; other documents into annual-report parts
(`AR:management`, `AR:risk`, ...), see :mod:`aidisclose.text.sections`.

**`LocalSource.can_verify_input_content(self) -> 'bool'`**

Do not bypass a subclass's custom fetch_document implementation.

**`LocalSource.config(self) -> 'dict'`**

Return the non-secret source settings used for provenance.

**`LocalSource.fetch_document(self, filing: 'Filing') -> 'Document'`**

Download (or read) a filing and return its extracted text.

**`LocalSource.fetch_document_verified(self, filing: 'Filing', expected_sha256: 'str | None') -> 'Document'`**

Check the exact bytes analysed against the run-start manifest.

**`LocalSource.input_content_path(self, filing: 'Filing') -> 'Path'`**

Resolve the same file path used for reading and checkpoint hashing.

**`LocalSource.list_filings(self, **_ignored) -> 'list[Filing]'`**

Return metadata for the filings matching *query*.

### `register_source`

```python
function register_source(name: 'str', cls: 'type[Source]') -> 'None'
```

Register a custom adapter under *name*.

Registered adapters can be used with :func:`get_source` and, when the
registering module is loaded with `aidisclose --plugin <module>` or
exposed through the `aidisclose.sources` entry-point group, from the CLI.

## EDGAR helpers

`aidisclose.sources.edgar`

### `country_from_edgar`

```python
function country_from_edgar(code: 'str | None') -> 'str | None'
```

Map an EDGAR state/country code to an ISO 3166-1 alpha-2 country code.

### `parse_filing_index`

```python
function parse_filing_index(html: 'str | bytes') -> 'list[dict[str, str]]'
```

Parse an EDGAR filing index page into `[{seq, description, document, type, url}]`.

## Text

`aidisclose.text`

### `extract_text`

```python
function extract_text(data: 'bytes | str', filename: 'str' = '', content_type: 'str' = '', reflow: 'bool' = True) -> 'str'
```

Dispatch to the right extractor based on file name, content type or content.

### `html_to_text`

```python
function html_to_text(html: 'str | bytes', drop_numeric_tables: 'bool' = True, drop_hidden: 'bool' = True, remove_artifacts: 'bool' = True, reflow: 'bool' = True) -> 'str'
```

Convert an HTML or inline-XBRL document to plain text (see module docstring).

### `split_sentences`

```python
function split_sentences(text: 'str', max_chars: 'int' = 1200) -> 'list[Span]'
```

Split text into sentences, never crossing line (block) boundaries.

### `split_sections`

```python
function split_sections(text: 'str', form: 'str | None' = None) -> 'dict[str, str]'
```

Split by Item schema for SEC forms, otherwise by annual-report parts.

### `split_items`

```python
function split_items(text: 'str', form: 'str' = '10-K', min_items: 'int' = 3) -> 'dict[str, str]'
```

Split an Item-structured SEC form into `{"<form>:<item>": text}`.

Returns `{"full": text}` when fewer than `min_items` distinct headings
are found. Text before the first selected heading (cover page, table of
contents, forward-looking statements) is returned as `"<form>:cover"`.
Combined headings such as "Items 1 and 2" are labelled with the first Item.

### `split_annual_report`

```python
function split_annual_report(text: 'str', min_words: 'int' = 30, min_parts: 'int' = 2) -> 'dict[str, str]'
```

Split a free-form annual report into `{"AR:<part>": text}`.

Parts: `front` (text before the first recognised heading),
`management`, `risk`, `sustainability`, `governance`,
`financial_statements` and `auditor`. Non-contiguous blocks of the same
part are concatenated in document order. Returns `{"full": text}` when
fewer than `min_parts` headings are recognised.

### `SectionFilter`

```python
class SectionFilter(sections=None)
```

Decide which section labels a pipeline processes.

A filter restricts only the document types it mentions: with
`["10-K:1A"]` a 10-K is limited to Item 1A while 20-F filings and
annual reports are processed in full. Unsplit documents (`"full"`) and
the main text of filings whose Items could not be found (`"<form>:main"`)
are always processed. Exhibit 13 (`"10-K:EX-13"`), which carries MD&A
incorporated by reference, is selected whenever `"10-K:7"` is.
`None` or an empty list selects everything. Preset names from
:data:`SECTION_PRESETS` (`"core"`, `"narrative"`) expand to their
labels and can be mixed with labels, e.g. `["core", "AR:sustainability"]`.

### `detect_language`

```python
function detect_language(text: 'str', min_tokens: 'int' = 30, min_score: 'float' = 0.15) -> 'tuple[str | None, dict[str, float]]'
```

Return `(language_code_or_None, scores)`.

## Lexicons

`aidisclose.lexicon`

### `Lexicon`

```python
class Lexicon(name: 'str', version: 'str', terms: 'Iterable[Term]', language: 'str' = 'en', description: 'str' = '', sha256: 'str' = '', source: 'str' = '') -> 'None'
```

A compiled set of AI terms.

**Parameters**

name, version, language:
    Identification metadata copied from the YAML file.
terms:
    Terms to match.
sha256:
    Digest of the source file (empty for programmatically built lexicons).

**`Lexicon.find(self, sentence: 'str', disabled_terms: 'Iterable[str]' = ()) -> 'LexiconHits'`**

Match all terms in *sentence* and apply disambiguation rules.

`disabled_terms` lists term ids that must be treated as false
positives for this sentence (used for document-level acronym
definitions such as "Artificial Insemination (AI)").

**`Lexicon.load(name_or_path: 'str | Path' = 'default', tiers: 'Iterable[str] | None' = ('core',)) -> 'Lexicon'`**

Load a built-in lexicon by name or a YAML file by path.

`tiers` restricts the terms to the given tiers (`None` keeps all).

### `conflicting_acronyms`

```python
function conflicting_acronyms(text: 'str') -> 'set[str]'
```

Return lexicon term ids whose acronym the document defines as a non-AI concept.

Example: a filing that writes 'artificial insemination ("AI")' and never
defines AI as artificial intelligence returns `{"ai_acronym"}`.

## Backends

`aidisclose.classify`

### `get_classifier`

```python
function get_classifier(name: 'str', **options: 'Any') -> 'Classifier'
```



### `RuleClassifier`

```python
class RuleClassifier(window: 'int' = 110, inherit_focal_from_context: 'bool' = True)
```

Deterministic, auditable baseline classifier.

**Parameters**

window:
    Number of characters on each side of an AI term in which deployment,
    specificity and context cues are considered to refer to that term.
inherit_focal_from_context:
    If a sentence has no agent cue, is written in the passive voice (or
    starts with "This/It/The system ...") and the previous sentence is in
    the focal firm's voice, treat the sentence as focal (lower confidence).

**`RuleClassifier.classify(self, candidates: 'Sequence[Candidate]') -> 'list[Prediction]'`**

Return one :class:`Prediction` per candidate, in order.

**`RuleClassifier.config(self) -> 'dict'`**

Configuration recorded in result provenance.

## LLM backend

`aidisclose.classify.llm`

### `LLMClassifier`

```python
class LLMClassifier(model: 'str | None' = None, base_url: 'str | None' = None, api_key: 'str | None' = None, temperature: 'float' = 0.0, seed: 'int' = 0, n_votes: 'int' = 1, timeout: 'float' = 120.0, max_retries: 'int' = 3, cache_path: 'str | Path | None' = '.aidisclose_cache/llm.sqlite', json_mode: 'bool' = True, max_tokens: 'int' = 200, session: 'requests.Session | None' = None, prompt: 'str' = 'codebook-v1.1', response_format: 'str' = 'auto', model_revision: 'str | None' = None, concurrency: 'int' = 1, max_target_chars: 'int' = 1500, max_context_chars: 'int' = 500) -> 'None'
```

Classify candidates with a chat model.

**Parameters**

model:
    Model name as known to the server (e.g. `"qwen2.5:14b-instruct"`).
base_url:
    Server root including `/v1` (default: `$AIDISCLOSE_LLM_BASE_URL` or
    `http://localhost:11434/v1`, Ollama's OpenAI-compatible endpoint).
api_key:
    Sent as a Bearer token if given (default: `$AIDISCLOSE_LLM_API_KEY`).
prompt:
    Prompt version from :data:`PROMPTS` (default `codebook-v1.1`).
cache_path:
    SQLite cache file (default: `.aidisclose_cache/llm.sqlite`).
n_votes:
    Number of sampled answers per sentence; the majority state is kept.
response_format:
    `"auto"` (JSON schema, then JSON mode, then plain text),
    `"json_schema"`, `"json_object"` or `"none"`.
json_mode:
    Deprecated compatibility switch; `False` means `response_format="none"`.
concurrency:
    Number of requests sent in parallel.
max_target_chars, max_context_chars:
    Truncation limits for the target sentence and each context sentence.

**`LLMClassifier.classify(self, candidates: 'Sequence[Candidate]') -> 'list[Prediction]'`**

Return one :class:`Prediction` per candidate, in order.

**`LLMClassifier.config(self) -> 'dict[str, Any]'`**

Configuration recorded in result provenance.

**`LLMClassifier.estimate(self, candidates: 'Sequence[Candidate]', chars_per_token: 'float' = 4.0) -> 'dict[str, Any]'`**

Requests and approximate tokens needed to label *candidates*.

Cached answers are not counted. Token counts use `chars_per_token`
(about 4 for English with common tokenizers) and are rough.

### `parse_response`

```python
function parse_response(content: 'str') -> 'tuple[AdoptionState, Reason, str]'
```

Extract state, reason and evidence from a model reply (tolerant to extra text).

## Supervised backend

`aidisclose.classify.supervised`

### `SupervisedClassifier`

```python
class SupervisedClassifier(pipeline=None, meta: 'dict[str, Any] | None' = None)
```

A backend that assigns an adoption state to candidate sentences.

**`SupervisedClassifier.classify(self, candidates: 'Sequence[Candidate]') -> 'list[Prediction]'`**

Return one :class:`Prediction` per candidate, in order.

**`SupervisedClassifier.config(self) -> 'dict[str, Any]'`**

Configuration recorded in result provenance.

**`SupervisedClassifier.cross_validate(candidates: 'Sequence[Candidate]', labels: 'Sequence[str]', k: 'int' = 5, random_state: 'int' = 0, C: 'float' = 4.0, weights: 'Sequence[float] | None' = None, strata: 'Sequence[str] | None' = None, groups: 'Sequence[str] | None' = None, n_boot: 'int' = 1000) -> 'dict[str, Any]'`**

Stratified k-fold estimate of out-of-sample performance.

Every sentence is predicted by a model that did not see it. The report
has the same fields as :func:`~aidisclose.validate.classification_report`
(sampling-weighted if `weights` are given), bootstrap confidence
intervals under `"ci"` and the out-of-fold predictions under
`"oof_predictions"`.

**`SupervisedClassifier.load(path: 'str | Path') -> 'SupervisedClassifier'`**

Load a model saved with :meth:`save`.

Model files are pickles: only load files you created or trust.

## Transformer backend (experimental)

`aidisclose.classify.transformer`

### `TransformerClassifier`

```python
class TransformerClassifier(model_path: 'str | Path', device: 'str | None' = None, batch_size: 'int' = 16, max_length: 'int' = 256)
```

Classify with a fine-tuned sequence-classification model (experimental).

**Parameters**

model_path:
    Directory written by :meth:`train` or any Hugging Face model whose
    `id2label` are the three states.
device:
    `"cpu"`, `"cuda"` ... (default: CUDA if available).

**`TransformerClassifier.classify(self, candidates: 'Sequence[Candidate]') -> 'list[Prediction]'`**

Return one :class:`Prediction` per candidate, in order.

**`TransformerClassifier.config(self) -> 'dict[str, Any]'`**

Configuration recorded in result provenance.

**`TransformerClassifier.train(candidates: 'Sequence[Candidate]', labels: 'Sequence[str]', out_dir: 'str | Path', base_model: 'str' = 'microsoft/deberta-v3-base', epochs: 'int' = 3, learning_rate: 'float' = 2e-05, batch_size: 'int' = 16, max_length: 'int' = 256, seed: 'int' = 42, device: 'str | None' = None) -> 'Path'`**

Fine-tune *base_model* on annotated candidates and save it to *out_dir*.

A plain PyTorch loop (AdamW, linear decay, no early stopping); the
training settings and a digest of the training data are written to
`aidisclose_training.json` next to the model.

## Validation: sampling and metrics

`aidisclose.validate`

### `sample_for_annotation`

```python
function sample_for_annotation(sentences: 'pd.DataFrame', n: 'int' = 400, strata: 'Iterable[str]' = ('state',), min_per_stratum: 'int' = 20, seed: 'int' = 42, include_false_positives: 'bool' = True) -> 'pd.DataFrame'
```

Draw a stratified random sample and attach sampling weights.

Allocation is proportional to stratum size with a floor of
`min_per_stratum` (so that rare states such as SUBSTANTIVE are well
represented). Each sampled row receives `sampling_weight` = stratum
size / sampled rows in the stratum, which lets
:func:`~aidisclose.validate.metrics.classification_report` estimate
population-level accuracy.

### `blind`

```python
function blind(sample: 'pd.DataFrame') -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Split a sample into an annotator file (no predictions) and a key file.

### `classification_report`

```python
function classification_report(y_true: 'Sequence[str]', y_pred: 'Sequence[str]', labels: 'Sequence[str]' = ['SUBSTANTIVE', 'WEAK', 'NO_CLAIM'], weights: 'Sequence[float] | None' = None) -> 'dict[str, Any]'
```

Per-class precision/recall/F1, macro-F1, accuracy, kappa and two binary views.

The binary views answer the questions that matter for panel indicators:
*substantive vs. rest* (`substantive_any`) and *claim vs. no claim*
(`claim_any`).

### `bootstrap_report`

```python
function bootstrap_report(y_true: 'Sequence[str]', y_pred: 'Sequence[str]', labels: 'Sequence[str]' = ['SUBSTANTIVE', 'WEAK', 'NO_CLAIM'], weights: 'Sequence[float] | None' = None, strata: 'Sequence[str] | None' = None, clusters: 'Sequence[str] | None' = None, n_boot: 'int' = 1000, alpha: 'float' = 0.05, seed: 'int' = 0) -> 'dict[str, tuple[float, float, float]]'
```

Percentile bootstrap CIs for every scalar in :func:`classification_report`.

Rows are resampled with replacement (within `strata` if given). When
`clusters` is supplied, whole clusters are sampled together and this
takes precedence over row strata. Sampling weights travel with each row.
Returns `{metric: (point, lower, upper)}`.

### `cohen_kappa`

```python
function cohen_kappa(a: 'Sequence[str]', b: 'Sequence[str]', labels: 'Sequence[str] | None' = None, weights: 'Sequence[float] | None' = None) -> 'float'
```

Cohen's (1960) kappa for two raters; `weights` are sampling weights, not disagreement weights.

## Validation: annotation files and agreement

`aidisclose.validate.annotation`

### `write_annotation_file`

```python
function write_annotation_file(df: 'pd.DataFrame', path: 'str | Path') -> 'Path'
```

Write an annotation sheet as `.xlsx` (with drop-downs) or `.csv` (UTF-8 with BOM).

### `read_annotation_file`

```python
function read_annotation_file(path: 'str | Path') -> 'pd.DataFrame'
```

Read an annotation sheet written by :func:`write_annotation_file` (or edited by hand).

### `to_label_studio`

```python
function to_label_studio(df: 'pd.DataFrame') -> 'list[dict[str, Any]]'
```

Convert an annotation sample to Label Studio tasks (import as JSON).

### `from_label_studio`

```python
function from_label_studio(export: 'str | Path | bytes | list') -> 'pd.DataFrame'
```

Read a Label Studio JSON export into `sentence_id, gold_state, gold_reason, annotator, notes`.

One row per non-cancelled annotation, so tasks labelled by two annotators
yield two rows (use :func:`agreement_report` on them).

### `agreement_report`

```python
function agreement_report(annotations: 'Sequence[pd.DataFrame] | pd.DataFrame', on: 'str' = 'sentence_id', column: 'str' = 'gold_state', annotator_column: 'str' = 'annotator') -> 'dict[str, Any]'
```

Inter-annotator agreement on `column`.

Pass either a list of annotation tables (one per annotator) or one long
table with an `annotator` column (e.g. from :func:`from_label_studio`).

### `fleiss_kappa`

```python
function fleiss_kappa(ratings: 'pd.DataFrame') -> 'float'
```

Fleiss' kappa for items (rows) rated by the same number of annotators (columns).

### `krippendorff_alpha_nominal`

```python
function krippendorff_alpha_nominal(ratings: 'pd.DataFrame') -> 'float'
```

Krippendorff's alpha for nominal data; rows are items, columns annotators, NaN = missing.

## Validation: recall audit

`aidisclose.validate.recall`

### `sample_recall`

```python
function sample_recall(pool: 'pd.DataFrame', n: 'int' = 300, share_probe: 'float' = 0.5, seed: 'int' = 42) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Draw `n` pool sentences for annotation; returns `(to_annotate, key)`.

`share_probe` of the sample comes from the probe stratum (or all of it if
it is smaller). The key file carries `recall_weight`, the inverse of the
overall inclusion probability (document stage x sample stage).

### `estimate_recall`

```python
function estimate_recall(recall_gold: 'pd.DataFrame', main_gold: 'pd.DataFrame | None' = None, found_substantive: 'float | None' = None, found_claim: 'float | None' = None, n_boot: 'int' = 1000, seed: 'int' = 0, alpha: 'float' = 0.05, found_source: 'str | None' = None) -> 'dict[str, Any]'
```

Estimate missed claims and lexicon recall.

**Parameters**

recall_gold:
    Annotated recall sample merged with its key (needs `gold_state`,
    `recall_weight` and `recall_stratum`).
main_gold:
    Annotated precision sample merged with its key (`gold_state`,
    `sampling_weight`, optional `sampling_stratum`), drawn from *all*
    retrieved sentences. It estimates how many retrieved sentences are
    truly SUBSTANTIVE (or claims).
found_substantive, found_claim:
    Alternatively, the numbers of retrieved true SUBSTANTIVE / claim
    sentences (e.g. predicted counts, which assumes the classifier's
    errors cancel; the report says which source was used).
found_source:
    Description of supplied counts; use an explicit prediction label when
    they are machine-generated. Not used with a main annotated sample.

## Validation: external benchmarks

`aidisclose.validate.external`

### `eurostat_ai_share`

```python
function eurostat_ai_share(path: 'str | Path | None' = None, size_emp: 'str' = 'GE250', indic_is: 'str' = 'E_AI_TANY', unit: 'str' = 'PC_ENT', nace_r2: 'str | None' = 'C10-S951_X_K', session=None) -> 'pd.DataFrame'
```

Share of enterprises using at least one AI technology, by country and year.

Returns `country` (ISO alpha-2), `year`, `eurostat_share` (a fraction)
and the selected `nace_r2`. The default selects the explicit activity
aggregate `C10-S951_X_K`; choose a different aggregate deliberately.

### `eurostat_ai_share_by_industry`

```python
function eurostat_ai_share_by_industry(path: 'str | Path | None' = None, indic_is: 'str' = 'E_AI_TANY', unit: 'str' = 'PC_ENT', countries: 'Sequence[str] | None' = None, session=None) -> 'pd.DataFrame'
```

Share of enterprises using AI by country, NACE section and year (dataset `isoc_eb_ain2`).

Eurostat publishes this breakdown for enterprises with 10 or more persons
employed only (`size_emp=GE10`) and for the NACE sections C, D, E, F, G,
H, I, J, L, M and N (no financial sector K). Sub-section and aggregate codes
(e.g. `C10-C12`, `L_M`) are dropped. Returns `country`,
`nace_section`, `year` and `eurostat_share` (fraction).

### `load_benchmark`

```python
function load_benchmark(path: 'str | Path', value: 'str', keys: 'Sequence[str]' = ('country', 'year'), rename: 'Mapping[str, str] | None' = None, percent: 'bool | None' = None, value_name: 'str' = 'benchmark_share') -> 'pd.DataFrame'
```

Load any external benchmark from a CSV file (e.g. a national survey of AI use).

`rename` maps the file's column names to AIDisclose keys (for example
`{"ref_year": "year", "naics_sector": "nace_section"}`; industry codes
must already be mapped to the classification of the panel). Values are
converted to fractions; `percent=None` treats them as percentages when
a value contains a percent sign or any value exceeds 1. ISO country codes
are upper-cased. Values at or below 1 without a percent sign are ambiguous;
set `percent=True` for percentages expressed in that form.

### `compare_with_external`

```python
function compare_with_external(group: 'pd.DataFrame', external: 'pd.DataFrame', measures: 'Sequence[str]' = ('share_keyword', 'share_claim', 'share_substantive'), group_keys: 'Sequence[str]' = ('country', 'fiscal_year'), external_keys: 'Sequence[str]' = ('country', 'year'), external_value: 'str' = 'eurostat_share', year_lag: 'int' = 0, min_firms: 'int' = 1) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Join group-level indicators with an external benchmark and correlate them.

`year_lag` shifts the disclosure year before matching (e.g. `1` compares
fiscal year *t* reports with survey year *t+1*).
Returns `(merged, summary)` where `summary` has one row per measure with
Spearman and Pearson correlations, a permutation p-value and the mean
absolute difference in percentage points.

## Quality checks

`aidisclose.qa`

### `quality_report`

```python
function quality_report(results, min_words: 'int' = 2000, long_sentence_chars: 'int' = 1000, duplicate_min_filings: 'int' = 5) -> 'dict[str, Any]'
```

Run the checks and return `{"summary": ..., "findings": [...]}`.

## Data model

`aidisclose.models`

### `AdoptionState`

```python
class AdoptionState
```

Three-state taxonomy for AI-related statements in corporate text.

SUBSTANTIVE
    The focal (reporting) firm states that it currently or previously
    uses, deploys, sells or has invested in a concrete AI system,
    product or capability, with at least one specific element
    (application, product, quantity or investment).
WEAK
    The focal firm refers to AI in a hedged, forward-looking,
    exploratory, risk-boilerplate or unspecific way.
NO_CLAIM
    The sentence contains no valid focal-firm claim about AI adoption:
    it is about third parties, markets or regulation, it negates
    adoption, or the keyword is a false positive.

### `Reason`

```python
class Reason
```

Reason codes that explain why a state was assigned.

### `Filing`

```python
class Filing(source: 'str', filing_id: 'str', firm_id: 'str', firm_name: 'str', fiscal_year: 'int | None' = None, period_end: 'str | None' = None, country: 'str | None' = None, industry_code: 'str | None' = None, industry_scheme: 'str | None' = None, form: 'str | None' = None, url: 'str | None' = None, language: 'str | None' = None, extra: 'dict[str, Any]' = <factory>) -> None
```

Metadata describing one disclosure document, independent of its source.

### `Document`

```python
class Document(filing: 'Filing', text: 'str', sections: 'dict[str, str]' = <factory>, language: 'str | None' = None, language_scores: 'dict[str, float]' = <factory>) -> None
```

Extracted plain text of a filing, optionally split into sections.

### `Candidate`

```python
class Candidate(filing_id: 'str', sentence_id: 'str', section: 'str', text: 'str', prev_text: 'str' = '', next_text: 'str' = '', firm_name: 'str' = '', terms: 'list[str]' = <factory>, term_spans: 'list[tuple[int, int]]' = <factory>, categories: 'list[str]' = <factory>, raw_terms: 'list[str]' = <factory>, filtered: 'list[str]' = <factory>, filter_reasons: 'dict[str, str]' = <factory>, baseline_terms: 'list[str]' = <factory>) -> None
```

A sentence that contains at least one AI lexicon term.

### `Prediction`

```python
class Prediction(state: 'AdoptionState', reason: 'Reason', confidence: 'float' = 1.0, backend: 'str' = '', details: 'dict[str, Any]' = <factory>) -> None
```

Label assigned to a candidate by a classifier backend.
