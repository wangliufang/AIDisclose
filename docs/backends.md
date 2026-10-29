# Classifier backends

All backends label the same candidate sentences with the same three states
and follow the same [codebook](codebook.md). Whatever the backend, its
accuracy on your corpus is unknown until you annotate a sample
([Validation](validation.md)); report the validation results of the backend
you use.

| Backend | When to use | Requirements | Reasons |
|---|---|---|---|
| `rules` (default) | transparent baseline; every label lists the cues that produced it | none | yes |
| `supervised` | after annotating a few hundred sentences | `aidisclose[ml]` | no |
| `llm` | a locally served open-weight model behind an OpenAI-compatible API | a running server | yes |
| `transformer` | **experimental**: a sequence classifier fine-tuned on your annotations | `aidisclose[transformers]` | no |

## Rules (rules 1.1.0)

For every lexicon hit the rules find the *agent* (the focal firm, a third
party or nobody), screen for negation, regulation and market-trend context,
look for deployment, product and investment cues, and require a specific
element (application, quantity, date, named product or model, investment)
before labelling a focal claim SUBSTANTIVE. All cues that fired are stored in
`details`, so every label can be audited. The rules are deterministic and need
no downloads. Known limitations and illustrative sentence tests are
described in the codebook. These tests are not an independent estimate of
accuracy on real reports.

## Supervised

```bash
aidisclose train --gold annotated.xlsx --key to_annotate_key.csv --results results/ --weighted --cv 5 --model-out model.joblib
aidisclose run ... --backend supervised --model-path model.joblib
```

Word and character n-gram TF-IDF of the sentence, plus tokens for the
section and for the cues the rule backend found (agent, forward-looking,
licensing in, ...), fed to a class-balanced logistic regression. Cross-validation reports out-of-fold, sampling-weighted
metrics, and `--weighted` applies the same sampling weights to the final model fit; evaluate the final model on a separate annotated sample.

With a sample key, `--results` must identify the original result dataset. The
command verifies the dataset digest and reconstructs candidates from recorded
terms, positions and context; baseline-only and filtered rows are excluded with
a count. Cross-validation describes the classifier-eligible sampled subset,
not the end-to-end pipeline's population performance. Weighted training requires
a complete annotation sample and complete classification. Without a key,
sentence-only matching is supported with a warning: document-level filtering
and original term positions cannot be established. Old results lacking positions
require the matching original lexicon and tiers or fresh extraction.

## LLM (prompt `codebook-v1.1`)

```bash
aidisclose run ... --backend llm --llm-model qwen2.5:14b-instruct \
    --llm-base-url http://localhost:11434/v1 --llm-concurrency 4
```

* **Prompt.** `codebook-v1.1` (default) states the codebook 1.1 definitions
  and decision rules, adds four worked examples (developer-written, not taken
  from any validation or stress set) and wraps the report text in
  `<previous>`, `<target>` and `<next>` tags that the model is told to treat
  as data, not instructions; tags inside the report text are neutralised.
  `--llm-prompt codebook-v1` selects the alternative frozen prompt; its
  digest and cache keys remain distinct.
* **Structured output.** `--llm-response-format auto` (default) asks for a
  JSON schema with the allowed states and reasons, falls back to JSON mode and
  then to plain text when the server rejects a format (HTTP 400/422), and
  keeps the format that worked for the rest of the run. Replies of reasoning
  models (`<think>...</think>`) are handled.
* **Robustness.** Long sentences and context are truncated
  (`max_target_chars`, `max_context_chars`) and the truncation is recorded in
  `details`; transient errors are retried with back-off; replies that cannot
  be parsed get `classification_status=error`, an empty state/reason, and are
  counted in `n_classifier_errors`. They are excluded from substantive and
  claim indicators, annotation samples and validation metrics; `--resume`
  retries those filings.
* **Throughput.** `--llm-concurrency N` sends N requests in parallel; combine
  with `--workers` for downloads.
* **Reproducibility.** Temperature 0 and a fixed seed by default; responses
  are cached in SQLite keyed by endpoint, model and declared model revision,
  prompt version, response format, decoding settings and input. Provenance
  records endpoint path, model revision, token limit, timeout and the response
  format used; `--llm-model-revision` can identify immutable weights. Use
  `--llm-votes 5 --llm-temperature 0.7` to sample several answers and
  stores the agreement rate. Every requested vote must succeed and there must be a strict majority, or the sentence is marked as an error.
* **Planning.** `aidisclose llm-estimate --results DIR --llm-model M` reports
  the number of requests (excluding cached answers) and an approximate token
  count before a run.

## Transformer (experimental)

```python
from aidisclose.classify.transformer import TransformerClassifier

TransformerClassifier.train(candidates, labels, out_dir="model/", base_model="microsoft/deberta-v3-base")
pipe = aid.Pipeline(backend="transformer", backend_options={"model_path": "model/"})
```

A plain PyTorch fine-tuning loop and batched inference; training settings and
a digest of the training data are saved next to the model. **Status:** this
code is exercised in the test suite with stand-in objects only; it has not
been benchmarked. Validate any model you train before using it.

## Relabelling an existing run

```bash
aidisclose relabel --results results/eu --backend llm --llm-model qwen2.5:14b-instruct --out results/eu_llm
```

`relabel` re-classifies the candidate sentences of a results directory with
another backend without downloading or segmenting the documents again, so
backends can be compared on identical candidates. The provenance records both
the new and the original classifier.

Transformer checkpoints are disabled because immutable model and tokenizer identity is not yet verified. Other transformer usage remains experimental.

## Term positions and relabelling

New result tables save the valid term offsets, so relabelling can reuse exactly the extracted positions without requiring the original lexicon file. Older tables without offsets require the original lexicon, selected tiers and document language; the tool verifies the stored lexicon digest before reconstructing positions. For example, pass `--lexicon path/to/original.yaml` and the original `--tiers`. Older programmatically constructed lexicons without an identifying digest cannot be verified: rerun extraction from the original documents. These checks do not authenticate deliberately modified files.
