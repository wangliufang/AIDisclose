# Getting started

## Install

```bash
# From the extracted AIDisclose folder
python -m pip install .                 # core (rules backend, Excel annotation sheets)
python -m pip install ".[ml,pdf]"       # + supervised backend and PDF reports
aidisclose doctor                       # check the installation
```

Python 3.10 or later is required. Dependencies are resolved by pip during installation.

## Try it offline

```bash
aidisclose demo
```

runs the full pipeline on a bundled **synthetic** corpus of 21 reports by
fictional companies (10-K, 20-F and European annual reports, one German
report that is skipped) and prints firm-year indicators, prevalence by year,
the evidence behind one indicator and a functional check against the labels
the demo sentences were written to illustrate. The demo is for testing only;
it is not data.

## A first study

```bash
# 1. list the sample once and keep the list
aidisclose list --source edgar --ticker MSFT JPM WMT SAP TM --forms 10-K 20-F --start 2018 --end 2025 \
    --user-agent "Jane Doe jane.doe@university.edu" --out filings.csv

# 2. process it (resumable, in parallel)
aidisclose run --source edgar --filings filings.csv --user-agent "Jane Doe jane.doe@university.edu" \
    --sections core --workers 4 --recall-random 5 --recall-probe-cap 10 --out results/

# 3. check the run
aidisclose qa --results results/

# 4. indicators
aidisclose panel --results results/ --level firm-year --out firm_year.csv
aidisclose panel --results results/ --level year --out year.csv

# 5. validate (see the Validation page)
aidisclose sample --results results/ --n 400 --min-per-stratum 40 --out to_annotate.xlsx
```

If step 2 is interrupted, run the same command with `--resume`.

## Python

```python
import aidisclose as aid

src = aid.EdgarSource(user_agent="Jane Doe jane.doe@university.edu")
filings = src.list_filings(tickers=["MSFT", "SAP"], forms=("10-K", "20-F"), start_year=2019, end_year=2024)

pipe = aid.Pipeline(backend="rules", sections="core")
results = pipe.run(filings, source=src, workers=4, checkpoint_dir="results/checkpoint")
results.save("results/")

panel = aid.firm_year_panel(results)
yearly = aid.group_panel(panel, by=["fiscal_year"])
evidence = results.evidence(firm_id=panel.firm_id[0], fiscal_year=2023)
```
