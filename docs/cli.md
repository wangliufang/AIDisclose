# Command line

Generated from `aidisclose --help` by `scripts/gen_docs.py`.

```text
usage: aidisclose [-h] [--version] [-v] [--plugin MODULE]
                  {demo,list,run,relabel,llm-estimate,panel,sample,sample-recall,evaluate-recall,agreement,import-label-studio,evaluate,train,eurostat,benchmark,qa,doctor,lexicon}
                  ...

Measure substantive AI adoption in corporate disclosures.

positional arguments:
  {demo,list,run,relabel,llm-estimate,panel,sample,sample-recall,evaluate-recall,agreement,import-label-studio,evaluate,train,eurostat,benchmark,qa,doctor,lexicon}
    demo                run the pipeline on the bundled synthetic corpus
                        (offline)
    list                list filings from a source
    run                 fetch filings and classify AI sentences
    relabel             re-classify the sentences of a results directory with
                        another backend
    llm-estimate        count LLM requests and tokens needed to label a
                        results directory
    panel               aggregate results into firm-year or group-year
                        indicators
    sample              draw a stratified sample for human annotation
    sample-recall       draw non-retrieved sentences from the recall pool for
                        annotation
    evaluate-recall     estimate missed claims and lexicon recall
    agreement           inter-annotator agreement (kappa, alpha)
    import-label-studio
                        convert a Label Studio JSON export to an annotation
                        CSV
    evaluate            compare annotations with predictions
    train               train the supervised backend on annotations
    eurostat            correlate a country-year panel with Eurostat AI-use
                        shares
    benchmark           correlate a group panel with any external benchmark
                        (CSV)
    qa                  quality checks for a results directory
    doctor              check the installation and (optionally) access to the
                        data sources
    lexicon             inspect a lexicon or test a sentence

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose
  --plugin MODULE       import a module that registers custom sources
                        (repeatable)
```

## `aidisclose demo`

```text
usage: aidisclose demo [-h] [--out OUT]

options:
  -h, --help  show this help message and exit
  --out OUT
```

## `aidisclose list`

```text
usage: aidisclose list [-h] --source {edgar,esef,local,xbrl}
                       [--ticker [TICKER ...]] [--cik [CIK ...]]
                       [--forms [FORMS ...]] [--exhibits {auto,none,all}]
                       [--fiscal-year-rule {period_end_year,jan_may_previous}]
                       [--user-agent USER_AGENT] [--country [COUNTRY ...]]
                       [--lei [LEI ...]] [--max-pages MAX_PAGES] [--dir DIR]
                       [--metadata METADATA] [--start START] [--end END]
                       [--cache-dir CACHE_DIR] [--source-option KEY=VALUE]
                       [--query KEY=VALUE] [--all-languages] --out OUT

options:
  -h, --help            show this help message and exit
  --all-languages       [xbrl] include non-English language hints
  --out OUT             output CSV

source:
  --source {edgar,esef,local,xbrl}
                        document source
  --ticker [TICKER ...]
                        [edgar] ticker symbols
  --cik [CIK ...]       [edgar] CIK numbers
  --forms [FORMS ...]   [edgar] 10-K, 20-F and/or 40-F (default 10-K)
  --exhibits {auto,none,all}
                        [edgar] add Exhibit 13 (10-K) and AIF/MD&A exhibits
                        (40-F); default auto
  --fiscal-year-rule {period_end_year,jan_may_previous}
                        [edgar] fiscal-year label convention
  --user-agent USER_AGENT
                        [edgar] 'Name email@domain' as required by the SEC
  --country [COUNTRY ...]
                        [xbrl] ISO country codes, e.g. DK SE GB
  --lei [LEI ...]       [xbrl] legal entity identifiers
  --max-pages MAX_PAGES
                        [xbrl] limit API pages (testing)
  --dir DIR             [local] folder with reports
  --metadata METADATA   [local] metadata CSV
  --start START         first fiscal year
  --end END             last fiscal year
  --cache-dir CACHE_DIR
                        cache for downloads (default .aidisclose_cache)
  --source-option KEY=VALUE
                        [custom sources] constructor argument (repeatable)
  --query KEY=VALUE     [custom sources] extra argument for list_filings
                        (repeatable)
```

## `aidisclose run`

```text
usage: aidisclose run [-h] --source {edgar,esef,local,xbrl}
                      [--ticker [TICKER ...]] [--cik [CIK ...]]
                      [--forms [FORMS ...]] [--exhibits {auto,none,all}]
                      [--fiscal-year-rule {period_end_year,jan_may_previous}]
                      [--user-agent USER_AGENT] [--country [COUNTRY ...]]
                      [--lei [LEI ...]] [--max-pages MAX_PAGES] [--dir DIR]
                      [--metadata METADATA] [--start START] [--end END]
                      [--cache-dir CACHE_DIR] [--source-option KEY=VALUE]
                      [--query KEY=VALUE]
                      [--backend {rules,supervised,llm,transformer}]
                      [--model-path MODEL_PATH] [--llm-model LLM_MODEL]
                      [--llm-model-revision LLM_MODEL_REVISION]
                      [--llm-base-url LLM_BASE_URL] [--llm-votes LLM_VOTES]
                      [--llm-max-tokens LLM_MAX_TOKENS]
                      [--llm-temperature LLM_TEMPERATURE]
                      [--llm-prompt LLM_PROMPT]
                      [--llm-response-format {auto,json_schema,json_object,none}]
                      [--llm-concurrency LLM_CONCURRENCY] [--lexicon LEXICON]
                      [--tiers [TIERS ...]] [--sections [SECTIONS ...]]
                      [--all-languages] [--languages [LANGUAGES ...]]
                      [--lexicon-for LANG=PATH]
                      [--recall-random RECALL_RANDOM]
                      [--recall-probe-cap RECALL_PROBE_CAP]
                      [--filings FILINGS] [--limit LIMIT] --out OUT
                      [--workers WORKERS] [--executor {thread,process}]
                      [--resume] [--no-checkpoint] [-q]

options:
  -h, --help            show this help message and exit
  --filings FILINGS     CSV produced by 'aidisclose list' (otherwise list on
                        the fly)
  --limit LIMIT         process only the first N filings
  --out OUT             output directory
  --workers WORKERS     concurrent workers (default 1)
  --executor {thread,process}
                        threads for downloads and LLM calls, processes for
                        CPU-bound local files
  --resume              continue an interrupted run in the same --out
                        directory
  --no-checkpoint       do not write per-filing checkpoints
  -q, --quiet

source:
  --source {edgar,esef,local,xbrl}
                        document source
  --ticker [TICKER ...]
                        [edgar] ticker symbols
  --cik [CIK ...]       [edgar] CIK numbers
  --forms [FORMS ...]   [edgar] 10-K, 20-F and/or 40-F (default 10-K)
  --exhibits {auto,none,all}
                        [edgar] add Exhibit 13 (10-K) and AIF/MD&A exhibits
                        (40-F); default auto
  --fiscal-year-rule {period_end_year,jan_may_previous}
                        [edgar] fiscal-year label convention
  --user-agent USER_AGENT
                        [edgar] 'Name email@domain' as required by the SEC
  --country [COUNTRY ...]
                        [xbrl] ISO country codes, e.g. DK SE GB
  --lei [LEI ...]       [xbrl] legal entity identifiers
  --max-pages MAX_PAGES
                        [xbrl] limit API pages (testing)
  --dir DIR             [local] folder with reports
  --metadata METADATA   [local] metadata CSV
  --start START         first fiscal year
  --end END             last fiscal year
  --cache-dir CACHE_DIR
                        cache for downloads (default .aidisclose_cache)
  --source-option KEY=VALUE
                        [custom sources] constructor argument (repeatable)
  --query KEY=VALUE     [custom sources] extra argument for list_filings
                        (repeatable)

classification:
  --backend {rules,supervised,llm,transformer}
  --model-path MODEL_PATH
                        [supervised, transformer] trained model file or
                        directory
  --llm-model LLM_MODEL
                        [llm] model name served by the endpoint
  --llm-model-revision LLM_MODEL_REVISION
                        [llm] immutable model revision or weights digest
  --llm-base-url LLM_BASE_URL
                        [llm] OpenAI-compatible base URL incl. /v1
  --llm-votes LLM_VOTES
                        [llm] sampled answers per sentence
  --llm-max-tokens LLM_MAX_TOKENS
                        [llm] maximum generated tokens per answer
  --llm-temperature LLM_TEMPERATURE
  --llm-prompt LLM_PROMPT
                        [llm] prompt version (codebook-v1.1, codebook-v1)
  --llm-response-format {auto,json_schema,json_object,none}
                        [llm] structured-output mode (auto falls back when the
                        server rejects one)
  --llm-concurrency LLM_CONCURRENCY
                        [llm] parallel requests
  --lexicon LEXICON     built-in lexicon name or YAML path
  --tiers [TIERS ...]   lexicon tiers (core, extended)
  --sections [SECTIONS ...]
                        section labels and/or presets to process, e.g. 'core'
                        or 10-K:1 10-K:1A 10-K:7 (default: all)
  --all-languages       do not skip non-English documents
  --languages [LANGUAGES ...]
                        languages to process (default en); others need
                        --lexicon-for
  --lexicon-for LANG=PATH
                        lexicon YAML for a language other than English
                        (repeatable)
  --recall-random RECALL_RANDOM
                        recall pool: random non-retrieved sentences per
                        document
  --recall-probe-cap RECALL_PROBE_CAP
                        recall pool: probe-term sentences per document
```

## `aidisclose relabel`

```text
usage: aidisclose relabel [-h] --results RESULTS [RESULTS ...]
                          [--backend {rules,supervised,llm,transformer}]
                          [--model-path MODEL_PATH] [--llm-model LLM_MODEL]
                          [--llm-model-revision LLM_MODEL_REVISION]
                          [--llm-base-url LLM_BASE_URL]
                          [--llm-votes LLM_VOTES]
                          [--llm-max-tokens LLM_MAX_TOKENS]
                          [--llm-temperature LLM_TEMPERATURE]
                          [--llm-prompt LLM_PROMPT]
                          [--llm-response-format {auto,json_schema,json_object,none}]
                          [--llm-concurrency LLM_CONCURRENCY]
                          [--lexicon LEXICON] [--tiers [TIERS ...]]
                          [--sections [SECTIONS ...]] [--all-languages]
                          [--languages [LANGUAGES ...]]
                          [--lexicon-for LANG=PATH]
                          [--recall-random RECALL_RANDOM]
                          [--recall-probe-cap RECALL_PROBE_CAP]
                          [--cache-dir CACHE_DIR] --out OUT

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
                        one or more results directories
  --cache-dir CACHE_DIR
                        cache for LLM responses
  --out OUT             output directory

classification:
  --backend {rules,supervised,llm,transformer}
  --model-path MODEL_PATH
                        [supervised, transformer] trained model file or
                        directory
  --llm-model LLM_MODEL
                        [llm] model name served by the endpoint
  --llm-model-revision LLM_MODEL_REVISION
                        [llm] immutable model revision or weights digest
  --llm-base-url LLM_BASE_URL
                        [llm] OpenAI-compatible base URL incl. /v1
  --llm-votes LLM_VOTES
                        [llm] sampled answers per sentence
  --llm-max-tokens LLM_MAX_TOKENS
                        [llm] maximum generated tokens per answer
  --llm-temperature LLM_TEMPERATURE
  --llm-prompt LLM_PROMPT
                        [llm] prompt version (codebook-v1.1, codebook-v1)
  --llm-response-format {auto,json_schema,json_object,none}
                        [llm] structured-output mode (auto falls back when the
                        server rejects one)
  --llm-concurrency LLM_CONCURRENCY
                        [llm] parallel requests
  --lexicon LEXICON     built-in lexicon name or YAML path
  --tiers [TIERS ...]   lexicon tiers (core, extended)
  --sections [SECTIONS ...]
                        section labels and/or presets to process, e.g. 'core'
                        or 10-K:1 10-K:1A 10-K:7 (default: all)
  --all-languages       do not skip non-English documents
  --languages [LANGUAGES ...]
                        languages to process (default en); others need
                        --lexicon-for
  --lexicon-for LANG=PATH
                        lexicon YAML for a language other than English
                        (repeatable)
  --recall-random RECALL_RANDOM
                        recall pool: random non-retrieved sentences per
                        document
  --recall-probe-cap RECALL_PROBE_CAP
                        recall pool: probe-term sentences per document
```

## `aidisclose llm-estimate`

```text
usage: aidisclose llm-estimate [-h] --results RESULTS [RESULTS ...]
                               [--backend {rules,supervised,llm,transformer}]
                               [--model-path MODEL_PATH]
                               [--llm-model LLM_MODEL]
                               [--llm-model-revision LLM_MODEL_REVISION]
                               [--llm-base-url LLM_BASE_URL]
                               [--llm-votes LLM_VOTES]
                               [--llm-max-tokens LLM_MAX_TOKENS]
                               [--llm-temperature LLM_TEMPERATURE]
                               [--llm-prompt LLM_PROMPT]
                               [--llm-response-format {auto,json_schema,json_object,none}]
                               [--llm-concurrency LLM_CONCURRENCY]
                               [--lexicon LEXICON] [--tiers [TIERS ...]]
                               [--sections [SECTIONS ...]] [--all-languages]
                               [--languages [LANGUAGES ...]]
                               [--lexicon-for LANG=PATH]
                               [--recall-random RECALL_RANDOM]
                               [--recall-probe-cap RECALL_PROBE_CAP]
                               [--cache-dir CACHE_DIR]

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
                        one or more results directories
  --cache-dir CACHE_DIR
                        cache for LLM responses

classification:
  --backend {rules,supervised,llm,transformer}
  --model-path MODEL_PATH
                        [supervised, transformer] trained model file or
                        directory
  --llm-model LLM_MODEL
                        [llm] model name served by the endpoint
  --llm-model-revision LLM_MODEL_REVISION
                        [llm] immutable model revision or weights digest
  --llm-base-url LLM_BASE_URL
                        [llm] OpenAI-compatible base URL incl. /v1
  --llm-votes LLM_VOTES
                        [llm] sampled answers per sentence
  --llm-max-tokens LLM_MAX_TOKENS
                        [llm] maximum generated tokens per answer
  --llm-temperature LLM_TEMPERATURE
  --llm-prompt LLM_PROMPT
                        [llm] prompt version (codebook-v1.1, codebook-v1)
  --llm-response-format {auto,json_schema,json_object,none}
                        [llm] structured-output mode (auto falls back when the
                        server rejects one)
  --llm-concurrency LLM_CONCURRENCY
                        [llm] parallel requests
  --lexicon LEXICON     built-in lexicon name or YAML path
  --tiers [TIERS ...]   lexicon tiers (core, extended)
  --sections [SECTIONS ...]
                        section labels and/or presets to process, e.g. 'core'
                        or 10-K:1 10-K:1A 10-K:7 (default: all)
  --all-languages       do not skip non-English documents
  --languages [LANGUAGES ...]
                        languages to process (default en); others need
                        --lexicon-for
  --lexicon-for LANG=PATH
                        lexicon YAML for a language other than English
                        (repeatable)
  --recall-random RECALL_RANDOM
                        recall pool: random non-retrieved sentences per
                        document
  --recall-probe-cap RECALL_PROBE_CAP
                        recall pool: probe-term sentences per document
```

## `aidisclose panel`

```text
usage: aidisclose panel [-h] --results RESULTS [RESULTS ...]
                        [--level {firm-year,country-year,industry-year,country-industry-year,year}]
                        [--min-firms MIN_FIRMS] [--industry-map INDUSTRY_MAP]
                        [--firm-attributes FIRM_ATTRIBUTES] [--weight WEIGHT]
                        [--repeat-threshold REPEAT_THRESHOLD]
                        [--exclude-unknown-years] [--confirm-xbrl-periods]
                        --out OUT [-q]

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
                        one or more results directories
  --level {firm-year,country-year,industry-year,country-industry-year,year}
  --min-firms MIN_FIRMS
  --industry-map INDUSTRY_MAP
                        CSV with industry_code,group to override NACE sections
  --firm-attributes FIRM_ATTRIBUTES
                        CSV with firm_id [,fiscal_year] and extra columns
  --weight WEIGHT       column for weighted group means (e.g. employees)
  --repeat-threshold REPEAT_THRESHOLD
                        similarity above which a claim repeats
  --exclude-unknown-years
                        exclude processed reports without a fiscal year (with
                        warning)
  --confirm-xbrl-periods
                        confirm included XBRL reports were manually verified
                        as annual
  --out OUT
  -q, --quiet
```

## `aidisclose sample`

```text
usage: aidisclose sample [-h] --results RESULTS [RESULTS ...] [--n N]
                         [--strata [STRATA ...]]
                         [--min-per-stratum MIN_PER_STRATUM] [--seed SEED]
                         --out OUT [--key KEY] [--label-studio LABEL_STUDIO]

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
                        one or more results directories
  --n N
  --strata [STRATA ...]
  --min-per-stratum MIN_PER_STRATUM
  --seed SEED
  --out OUT             annotation sheet without predictions (.xlsx or .csv)
  --key KEY             CSV with predictions and sampling weights
  --label-studio LABEL_STUDIO
                        also write Label Studio tasks (JSON) and config
```

## `aidisclose sample-recall`

```text
usage: aidisclose sample-recall [-h] --results RESULTS [RESULTS ...] [--n N]
                                [--share-probe SHARE_PROBE] [--seed SEED]
                                --out OUT [--key KEY]
                                [--label-studio LABEL_STUDIO]

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
                        results directories with recall_pool.csv
  --n N
  --share-probe SHARE_PROBE
                        share of the sample from the probe stratum
  --seed SEED
  --out OUT             annotation sheet (.xlsx or .csv)
  --key KEY             CSV with recall weights
  --label-studio LABEL_STUDIO
                        also write Label Studio tasks (JSON) and config
```

## `aidisclose evaluate-recall`

```text
usage: aidisclose evaluate-recall [-h] --gold GOLD --key KEY
                                  [--main-gold MAIN_GOLD]
                                  [--main-key MAIN_KEY]
                                  [--results [RESULTS ...]] [--n-boot N_BOOT]
                                  [--out OUT]

options:
  -h, --help            show this help message and exit
  --gold GOLD           annotated recall sample
  --key KEY             key file from 'sample-recall'
  --main-gold MAIN_GOLD
                        annotated precision sample (from 'sample')
  --main-key MAIN_KEY   key file from 'sample'
  --results [RESULTS ...]
                        fallback: use predicted SUBSTANTIVE counts
  --n-boot N_BOOT
  --out OUT             Markdown report path (a .json copy is written too)
```

## `aidisclose agreement`

```text
usage: aidisclose agreement [-h] [--annotations [ANNOTATIONS ...]]
                            [--label-studio LABEL_STUDIO] [--column COLUMN]
                            [--out OUT]

options:
  -h, --help            show this help message and exit
  --annotations [ANNOTATIONS ...]
                        one annotated file per annotator
  --label-studio LABEL_STUDIO
                        Label Studio JSON export with several annotators
  --column COLUMN       label column (default gold_state)
  --out OUT             Markdown report path (a .json copy is written too)
```

## `aidisclose import-label-studio`

```text
usage: aidisclose import-label-studio [-h] --export EXPORT --out OUT

options:
  -h, --help       show this help message and exit
  --export EXPORT
  --out OUT
```

## `aidisclose evaluate`

```text
usage: aidisclose evaluate [-h] --gold GOLD --results RESULTS [RESULTS ...]
                           [--key KEY] [--on ON [ON ...]] [--weighted]
                           [--cluster-by {filing_id,firm_id,none}]
                           [--n-boot N_BOOT] [--title TITLE] [--out OUT]

options:
  -h, --help            show this help message and exit
  --gold GOLD           annotated CSV with gold_state
  --results RESULTS [RESULTS ...]
                        one or more results directories
  --key KEY             key file from 'sample' (sampling weights and strata)
  --on ON [ON ...]      join columns (default sentence_id; required for
                        weighted evaluation)
  --weighted            use sampling weights (population estimates)
  --cluster-by {filing_id,firm_id,none}
                        bootstrap whole filings/firms for confidence intervals
                        (default filing_id)
  --n-boot N_BOOT
  --title TITLE
  --out OUT             Markdown report path (a .json copy is written too)
```

## `aidisclose train`

```text
usage: aidisclose train [-h] --gold GOLD --model-out MODEL_OUT [--cv CV]
                        [--cv-group-by {firm_id,filing_id,sentence}]
                        [--key KEY] [--results RESULTS [RESULTS ...]]
                        [--weighted] [--n-boot N_BOOT] [--oof-out OOF_OUT]
                        [--lexicon LEXICON] [--tiers [TIERS ...]]

options:
  -h, --help            show this help message and exit
  --gold GOLD
  --model-out MODEL_OUT
  --cv CV               also report k-fold cross-validation
  --cv-group-by {firm_id,filing_id,sentence}
                        keep firms or filings within one CV fold; sentence
                        reproduces random-sentence CV
  --key KEY             key file from 'sample'; requires matching --results
  --results RESULTS [RESULTS ...]
                        original results used to produce the sample
  --weighted            use sampling weights for training and cross-validation
  --n-boot N_BOOT
  --oof-out OOF_OUT     CSV with out-of-fold predictions
  --lexicon LEXICON     lexicon used by the pipeline
  --tiers [TIERS ...]   lexicon tiers used by the pipeline
```

## `aidisclose eurostat`

```text
usage: aidisclose eurostat [-h] --panel PANEL --eurostat-file EUROSTAT_FILE
                           [--dataset {isoc_eb_ai,isoc_eb_ain2}] [--size SIZE]
                           [--nace-aggregate NACE_AGGREGATE] [--lag LAG]
                           [--min-firms MIN_FIRMS] --out OUT

options:
  -h, --help            show this help message and exit
  --panel PANEL         country-year panel CSV ('aidisclose panel --level
                        country-year')
  --eurostat-file EUROSTAT_FILE
                        JSON-stat file of the dataset
  --dataset {isoc_eb_ai,isoc_eb_ain2}
                        isoc_eb_ai: country x size class; isoc_eb_ain2:
                        country x NACE section (10+ employees)
  --size SIZE           enterprise size: country default GE250; industry only
                        GE10
  --nace-aggregate NACE_AGGREGATE
                        [isoc_eb_ai] exact NACE activity aggregate (default
                        C10-S951_X_K)
  --lag LAG             survey year minus fiscal year
  --min-firms MIN_FIRMS
  --out OUT
```

## `aidisclose benchmark`

```text
usage: aidisclose benchmark [-h] --panel PANEL --file FILE --value VALUE
                            [--keys KEYS [KEYS ...]]
                            [--rename [FILE_COL=KEY ...]]
                            [--scale {auto,percent,fraction}] [--lag LAG]
                            [--min-firms MIN_FIRMS] --out OUT

options:
  -h, --help            show this help message and exit
  --panel PANEL         group panel CSV from 'aidisclose panel'
  --file FILE           benchmark CSV
  --value VALUE         benchmark column with the share or percentage
  --keys KEYS [KEYS ...]
                        join keys (year = fiscal year)
  --rename [FILE_COL=KEY ...]
                        rename benchmark columns
  --scale {auto,percent,fraction}
  --lag LAG             benchmark year minus fiscal year
  --min-firms MIN_FIRMS
  --out OUT
```

## `aidisclose qa`

```text
usage: aidisclose qa [-h] --results RESULTS [RESULTS ...]
                     [--min-words MIN_WORDS] [--out OUT] [--strict]

options:
  -h, --help            show this help message and exit
  --results RESULTS [RESULTS ...]
  --min-words MIN_WORDS
  --out OUT             Markdown report path (a .json copy is written too)
  --strict              exit with status 1 if there are warnings
```

## `aidisclose doctor`

```text
usage: aidisclose doctor [-h] [--network] [--user-agent USER_AGENT]
                         [--llm-base-url LLM_BASE_URL] [--cache-dir CACHE_DIR]

options:
  -h, --help            show this help message and exit
  --network             also test SEC, filings.xbrl.org and Eurostat
  --user-agent USER_AGENT
                        'Name email@domain' for the network checks
  --llm-base-url LLM_BASE_URL
                        also test an OpenAI-compatible endpoint
  --cache-dir CACHE_DIR
```

## `aidisclose lexicon`

```text
usage: aidisclose lexicon [-h] [--name NAME] [--tiers [TIERS ...]]
                          [--test TEST] [-q]

options:
  -h, --help           show this help message and exit
  --name NAME
  --tiers [TIERS ...]
  --test TEST          sentence to match
  -q, --quiet
```
