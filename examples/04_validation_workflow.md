# Case study 3: validating the classifier on your corpus

```bash
# 0. check the runs first
aidisclose qa --results case_edgar/results case_esef/results --out annotation/qa.md

# 1. draw one stratified sample from one or more results directories (Excel sheet with drop-downs)
aidisclose sample --results case_edgar/results case_esef/results --n 400 --min-per-stratum 40 --seed 2026 \
    --out annotation/sample.xlsx            # also writes annotation/sample_key.csv (predictions, weights, strata)

# 2. two annotators label copies of annotation/sample.xlsx independently (gold_state, gold_reason),
#    following annotation/codebook.md; measure agreement, then adjudicate into annotation/annotated.xlsx
aidisclose agreement --annotations annotation/sample_A.xlsx annotation/sample_B.xlsx --out annotation/agreement.md

# 3. rule backend: sampling-weighted metrics with bootstrap confidence intervals
aidisclose evaluate --gold annotation/annotated.xlsx --results case_edgar/results case_esef/results \
    --key annotation/sample_key.csv --weighted --out annotation/report_rules.md

# 4. recall audit: annotate sentences the lexicon did not retrieve
aidisclose sample-recall --results case_edgar/results case_esef/results --n 300 --out annotation/recall.xlsx
aidisclose evaluate-recall --gold annotation/recall_done.xlsx --key annotation/recall_key.csv \
    --main-gold annotation/annotated.xlsx --main-key annotation/sample_key.csv --out annotation/recall.md

# 5. supervised backend: weighted 5-fold cross-validation on the adjudicated labels
aidisclose train --gold annotation/annotated.xlsx --key annotation/sample_key.csv \
    --results case_edgar/results case_esef/results --weighted --cv 5 \
    --oof-out annotation/oof_supervised.csv --model-out models/supervised.joblib

# 6. language-model backend on exactly the same candidate sentences (local open-weight model)
aidisclose llm-estimate --results case_edgar/results --llm-model qwen2.5:14b-instruct
aidisclose relabel --results case_edgar/results --backend llm --llm-model qwen2.5:14b-instruct \
    --llm-base-url http://localhost:11434/v1 --llm-concurrency 4 --out case_edgar/results_llm
# The relabelled dataset has a different digest: draw and annotate a new sample.
aidisclose sample --results case_edgar/results_llm --out annotation/sample_llm.xlsx
# After annotation, save as annotation/annotated_llm.xlsx.
aidisclose evaluate --gold annotation/annotated_llm.xlsx --results case_edgar/results_llm \
    --key annotation/sample_llm_key.csv --weighted --out annotation/report_llm.md
```

Report, for every backend you use: the number of annotated sentences,
inter-annotator agreement, per-state precision/recall/F1 with bootstrap
confidence intervals, the recall audit, and the codebook, lexicon, rule and
prompt versions (all in `provenance.json`).

If annotators work in Label Studio instead, add `--label-studio
annotation/tasks.json` to `sample`, import the tasks with the generated
`annotation/tasks_config.xml` as the labelling interface, and convert the
export with `aidisclose import-label-studio --export export.json --out
annotation/annotated.csv` (one row per annotator; `aidisclose agreement
--label-studio export.json` computes agreement directly).

Sentence identifiers have the form `<filing id>#<section>#<sentence number>`.
They depend on the document and text extraction, rather than backend labels.
Weighted evaluation additionally binds each sample key to its result dataset;
a relabelled dataset therefore requires a new sample and key. `evaluate` checks
annotated text against predictions and reports excluded mismatches.

Evaluate the supervised backend only through cross-validation on the sample it
was trained on (step 5); to compare it on equal terms with other backends,
annotate a second sample and use `aidisclose relabel --backend supervised`.
