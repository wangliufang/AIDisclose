# Reproducibility and large runs

## Provenance

Each pipeline run writes `provenance.json` next to its results:

```json
{
  "aidisclose_version": "0.5.3",
  "created_utc": "...",
  "python": "3.12.4",
  "lexicon": {"name": "aidisclose-en", "version": "1.1.0[core]", "sha256": "..."},
  "baseline_lexicon": {"name": "keyword-baseline-4", "version": "1.0.0[all]", "sha256": "..."},
  "classifier": {"backend": "rules", "version": "1.1.0", "window": 110},
  "sections": ["10-K:1", "10-K:1A", "10-K:7"],
  "languages": ["en"],
  "recall_pool": {"random_per_document": 5, "probe_cap_per_document": 10, "seed": 20240601, "...": "..."}
}
```

The run also records an ordered input-manifest digest, filing count, source
configuration and installed dependency versions. Local file bytes contribute
to the manifest digest, including file URLs resolved by `LocalSource` when an
`extra.path` is absent. Built-in LocalSource verifies the exact bytes it reads
against the run-start digest and records a failed document if they differ or
were initially unavailable. This is not a filesystem snapshot or a lock.
Custom sources or LocalSource subclasses overriding fetch may lack this check;
verify their provenance and keep inputs stable. `run_documents` hashes the ordered filing metadata,
input text, ordered sections and declared language before processing; it does
not alter the caller's document language. The source-run manifest does not hash
the contents fetched from a remote URL, so preserve remote inputs separately.
For concatenated results, `combined_runs` records each constituent digest;
no constituent digest is presented as the combined input digest. Nested concatenations preserve the flattened original runs in order; differing relabelling timestamps alone do not make compatible configurations incompatible. The `classifier`
entry depends on the backend: the rule
backend records its version and settings; the supervised backend its training
metadata; the `llm` backend records a sanitized endpoint path (no credentials
or query string), model name and declared revision, decoding settings, token
limit, timeout, prompt digest and response mode. Each LLM prediction records
the format the endpoint accepted. `relabel` adds the original classifier.

## Measurement components

Record the software version together with the actual lexicon, rule, codebook,
prompt, model and benchmark settings used for each study. The provenance
record includes component identifiers and available hashes. The package also
contains a frozen lexicon and an alternative prompt for reproducibility; use
the exact settings that match your study.

## Caching

* Downloaded documents are cached under `.aidisclose_cache/<source>/` keyed
  by URL. Filing *lists* are fetched fresh by design (indexes change); save
  them with `aidisclose list --out filings.csv` and reuse them with
  `aidisclose run --filings filings.csv` to fix the sample. Archiving the
  cache together with the results lets others rerun a study on the same
  documents.
* Language-model responses are cached in `.aidisclose_cache/llm.sqlite` keyed
  by endpoint, model and declared revision, prompt version, response mode,
  decoding settings and input. Specify `--llm-model-revision` when a model
  name is a mutable alias; AIDisclose cannot infer a remote model's weight hash.

## Checkpoints and resume

`aidisclose run --out DIR` writes one checkpoint file per processed filing to
`DIR/checkpoint/` as it goes. After an interruption,

```bash
aidisclose run ... --out DIR --resume
```

skips filings already processed and retries failed downloads and classifier
errors. The configuration (including source settings) and ordered filing
manifest are fingerprinted; resuming with a different configuration or filing
list is refused. Local filing bytes are included in the manifest. A run without
`--resume` starts afresh. In Python:
`Pipeline.run(filings, source, checkpoint_dir=..., resume=True)`.

## Parallel runs

```bash
aidisclose run ... --workers 8                        # threads: downloads and LLM calls
aidisclose run --source local ... --workers 8 --executor process   # processes: CPU-bound extraction
```

For the synthetic local fixture, results match a sequential run in input order. Within one run, source HTTP clients share a request clock across process workers. Other runs and applications are not coordinated; live tests do not establish the safety of concurrent independent runs.

## Languages

AIDisclose ships English lexicons and English rule cues only. Documents in
other languages are skipped (`skipped_non_english`) and reported by
`aidisclose qa`, because a skipped share that differs between countries biases
country comparisons. For other languages, the pipeline can route documents to
a user-supplied lexicon (`--languages en de --lexicon-for de=lexicon_de.yaml`
with the `supervised` or `llm` backend); documents without a matching lexicon
are marked `skipped_no_lexicon` rather than silently routed to English, and
any supplied lexicon needs its own
validation, and none is distributed with the package.

## Sampling variability of language models

With `--llm-votes 5 --llm-temperature 0.7` each sentence is labelled five
times and the majority label is kept; the `agreement` field in `details`
shows how stable the label is. Low-agreement sentences are good candidates
for human review.

## Environment

`aidisclose doctor` prints the versions of Python and of all dependencies and
checks the cache directory; `aidisclose doctor --network --user-agent "..."`
also tests access to SEC EDGAR, filings.xbrl.org, Eurostat and (with
`--llm-base-url`) a language-model endpoint.

## Result, checkpoint and sample integrity

Saved result files include hashes and an interrupted-save marker. Loading
rejects missing or changed files, and results without provenance or an integrity
manifest are refused by default. Hashes and sample-key seals detect accidental
changes or mismatches; they are not signatures and cannot authenticate
deliberate edits. To inspect an unverified result in Python, use
`Results.load(path, allow_unverified=True)` only after independently checking
it; saving a new copy does not establish its original integrity.

Checkpoint metadata bind the input manifest and configuration, including the
software and classifier versions. Checkpoint parts are checked against their
filing identifiers and content digests; invalid parts are reprocessed.
Resume stops without deleting parts when metadata are missing. Use a new
checkpoint directory when the configuration or software version changes.
Cache writes and checkpoint metadata replacement are atomic, but existing
cache entries are not cryptographically authenticated.

Sample keys are bound to their result tables and carry a completeness seal.
They protect selected fields, including visible context when available.
Keep the original results and sampling keys together; regenerate and annotate
again when a key cannot be matched or verified. For older keys with absent
context fields, compare only available fields and retain the original context.
Concatenation rejects duplicate filing identifiers. Identifiers and industry
codes are preserved as text, including leading zeroes.

Weighted evaluation requires complete classification and correctly matched
annotations and weights. Sparse strata may not support bootstrap intervals.
Use unique file paths in local metadata; ambiguous basename matches are
rejected. For details of sampling and benchmarks, see `docs/validation.md`.

## Classifier and relabelling safeguards

A supervised model fingerprint covers fitted model state and relevant
parameters. A different model is refused for an existing checkpoint.
Transformer checkpointing is disabled until immutable model and tokenizer
identity is established; transformer inference without checkpoints remains
experimental. For LLM multi-vote classification, every requested response
must be valid and the leading label must have a strict majority; otherwise
the sentence is an error and is retried on resume. Invalid or state-incompatible
LLM reason codes are rejected; invalid cached replies are discarded.

Provenance includes a digest of active lexicon terms and the YAML file where
applicable. Relabelling reuses saved term positions; if they are absent, the
original lexicon, tier selection and document language must be identifiable
and verifiable. Otherwise re-extract the original documents.

Output paths must be distinct from identified input files and result folders.
The program does not automatically back up unrelated existing output files;
choose output locations carefully. XBRL source identifiers distinguish report
records, while the fiscal-year basis is recorded for SEC filings when a year
must be estimated from a filing date.
