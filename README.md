# AIDisclose

**Measure substantive AI adoption claims in corporate disclosures, beyond keyword counts.**

AIDisclose is an open-source Python package that retrieves corporate reports from
regulatory sources, finds sentences that mention artificial intelligence and
classifies each of them into one of three states:

| State | Meaning | Example |
|---|---|---|
| `SUBSTANTIVE` | the reporting firm uses, sells or has invested in a concrete AI system, with a specific element | "We use machine learning models to forecast freight demand at 85 depots." |
| `WEAK` | the firm's own AI is mentioned in a forward-looking, hedged, boilerplate or vague way | "We plan to use artificial intelligence to enhance our customer experience." |
| `NO_CLAIM` | no valid claim about the firm's own AI (third parties, markets, regulation, negation, false positives) | "Competitors may use AI to offer lower prices." |

The labels are aggregated into firm-year and country/industry-year indicators
that sit next to a plain keyword baseline, so researchers can see how much a
keyword measure overstates substantiated adoption. Every indicator can be
traced back to the sentences behind it.

* **Sources across jurisdictions**: SEC EDGAR (Forms 10-K, 20-F and 40-F, with
  Exhibit 13 and 40-F exhibits), filings.xbrl.org (ESEF and UKSEF annual
  reports) and any local folder of HTML/XHTML/PDF/TXT reports; new regulators
  are added with a small adapter class.
* **Sections**: 10-K and 20-F Items and the parts of European annual reports
  (management report, risks, sustainability statement, governance, ...), with
  section presets.
* **Four interchangeable classifiers**: transparent rules (default, no
  downloads), a supervised model trained on your own annotations, any
  OpenAI-compatible endpoint for locally served open-weight language models,
  and an experimental fine-tuned transformer.
* **Indicators**: prevalence and intensity by state, technique category and
  type of claim; novelty against the previous year's report; onset of
  substantive claims; firm-weighted or size-weighted group means.
* **Validation kit**: blind stratified samples as Excel sheets or Label Studio
  tasks, sampling-weighted metrics with bootstrap intervals, inter-annotator
  agreement (Cohen, Fleiss, Krippendorff), a **recall audit** of sentences the
  lexicon did not retrieve, and comparisons with Eurostat or any other
  benchmark.
* **Large, reproducible runs**: resumable checkpoints, parallel workers within
  the sources' rate limits, versioned lexicons and prompts with SHA-256
  digests, cached downloads and model responses, a provenance record for every
  run, and a `qa` report that flags silent failures.

Classifier failures are recorded as missing labels, excluded from adoption
indicators, and retried by `--resume`. Supervised cross-validation keeps each
firm or filing within one fold. Firm-year panels report classification
completeness so incomplete labels are not mistaken for non-adoption.

Version 0.5.3 processes **English-language** documents; other languages are
skipped and reported (the pipeline can route them to user-supplied lexicons,
but none is shipped).

## Installation

```bash
# From the extracted AIDisclose folder
python -m pip install .             # core
python -m pip install ".[ml,pdf]"   # + supervised backend and PDF reports
aidisclose doctor                   # check the installation
```

Python 3.10 or later is required.

## Try it in one minute (offline)

```bash
aidisclose demo
```

This runs the full pipeline on a bundled **synthetic** corpus of fictional
companies and prints firm-year indicators, prevalence by year, the evidence
sentences behind one indicator, and a functional check against the labels the
demo sentences were written to illustrate. The demo corpus is for testing only.

## Workflow

```bash
# 1. process 10-K and 20-F filings (the SEC requires a User-Agent with contact details)
aidisclose run --source edgar --ticker MSFT JPM SAP TM --forms 10-K 20-F --start 2018 --end 2025 \
    --user-agent "Jane Doe jane.doe@university.edu" --sections core --workers 4 \
    --recall-random 5 --recall-probe-cap 10 --out results/sec          # add --resume after an interruption

# 2. XBRL listings include interim reports; manually select verified annual filings
aidisclose list --source xbrl --country DK NL IE GB --start 2021 --end 2024 --out eu_listings.csv
# Review each report, retain one verified annual record per firm-year in eu_annual.csv.
aidisclose run --source xbrl --filings eu_annual.csv --sections core --out results/eu

# 3. any folder of reports, e.g. from another regulator
aidisclose run --source local --dir reports/ --metadata reports/metadata.csv --out results/local

# 4. check the run, then build indicators
aidisclose qa    --results results/eu
aidisclose panel --results results/eu --level firm-year --confirm-xbrl-periods --out eu_firm_year.csv
aidisclose panel --results results/eu --level country-year --confirm-xbrl-periods --out eu_country_year.csv

# 5. validation
aidisclose sample          --results results/eu --n 400 --out to_annotate.xlsx      # blind sheet + key file
aidisclose evaluate        --gold to_annotate.xlsx --results results/eu --key to_annotate_key.csv --weighted
aidisclose agreement       --annotations ann_a.xlsx ann_b.xlsx
aidisclose sample-recall   --results results/sec --n 300 --out recall.xlsx
aidisclose evaluate-recall --gold recall.xlsx --key recall_key.csv --main-gold to_annotate.xlsx \
                           --main-key to_annotate_key.csv
aidisclose eurostat        --panel eu_country_year.csv --eurostat-file isoc_eb_ai.json --out eurostat.csv
```

## Python API

```python
import aidisclose as aid

src = aid.EdgarSource(user_agent="Jane Doe jane.doe@university.edu")
filings = src.list_filings(tickers=["MSFT", "SAP"], forms=("10-K", "20-F"), start_year=2019, end_year=2024)

pipe = aid.Pipeline(backend="rules", sections="core")
results = pipe.run(filings, source=src, workers=4, checkpoint_dir="results/checkpoint")
results.save("results/")

panel = aid.firm_year_panel(results)                       # firm-year indicators
yearly = aid.group_panel(panel, by=["fiscal_year"])        # prevalence by year
evidence = results.evidence(firm_id=panel.firm_id[0], fiscal_year=2023)
```

## Classifier backends

| Backend | When to use | Requirements |
|---|---|---|
| `rules` | transparent baseline; every label lists the cues that produced it | none |
| `supervised` | after annotating a few hundred sentences; `aidisclose train --gold ... --cv 5` | `aidisclose[ml]` |
| `llm` | locally served open-weight model behind an OpenAI-compatible API (vLLM, Ollama, llama.cpp) | a running server |
| `transformer` | experimental: a sequence classifier fine-tuned on your annotations | `aidisclose[transformers]` |

```bash
aidisclose relabel --results results/eu --backend llm --llm-model qwen2.5:14b-instruct \
    --llm-base-url http://localhost:11434/v1 --llm-concurrency 4 --out results/eu_llm
```

All backends share the [annotation codebook](annotation/codebook.md). The
accuracy of any backend on your corpus is unknown until you annotate a sample:
always report validation results for the backend you use.

## Documentation

The `docs/` folder (also buildable as a site with `python scripts/gen_docs.py && mkdocs build`):

* [Getting started](docs/getting-started.md)
* [Sources and adapters](docs/sources.md) and [Sections](docs/sections.md)
* [Backends](docs/backends.md) and [Indicators](docs/indicators.md)
* [Validation](docs/validation.md): annotation, agreement, recall audit, benchmarks, quality checks
* [Reproducibility and large runs](docs/reproducibility.md)
* [Codebook](annotation/codebook.md) and [Examples](examples/)
* [XBRL annual-period review](docs/xbrl_period_review.md) and [real-filing example](docs/real_case.md)
* [中文使用说明](README_CN.md)

## Citing

See `CITATION.cff` for the available software attribution and version details.

## Authors

* **Liufang Wang** — College of Social Science, Yeungnam University, Gyeongsan-si, Gyeongbuk 38541, Republic of Korea. Email: qq1055145729@gmail.com
* **Xiangyang Sun**\* — College of Business Administration, Changshin University, 262 Paryong-ro, Masanhoewon-gu Changwon-si, Gyeongsangnam-do 51352, Republic of Korea. Email: sunx200002@gmail.com

\* Corresponding author.

## License

MIT. Downloaded filings remain subject to the terms of their sources.

## Important usage considerations

* Verify the reporting period of each XBRL filing and select one annual report
  per firm-year before aggregation. The confirmation option records your review;
  it does not verify a report automatically. See [the period guide](docs/xbrl_period_review.md).
* Classifier labels are reported claims, not proof of actual AI adoption.
  Independently annotate and evaluate a representative sample of your own corpus.
* Sampling and result integrity checks detect accidental mismatch or corruption;
  unkeyed hashes and seals are not cryptographic authentication. Preserve the
  original result files, sampling keys, inputs, and relevant provenance.
* Recall interval estimates have limitations; see [Validation](docs/validation.md)
  and [Reproducibility](docs/reproducibility.md) before interpreting results.
