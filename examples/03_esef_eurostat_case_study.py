"""Case study 2: English-language ESEF reports compared with Eurostat AI-use statistics.

Requires network access to filings.xbrl.org and ec.europa.eu:

    aidisclose list --source xbrl --country DK --start 2023 --end 2023 --out listings.csv
    # Review reports manually and keep only verified annual records in annual.csv.
    python examples/03_esef_eurostat_case_study.py --filings annual.csv --confirm-xbrl-periods

Outputs (in ./case_esef/): results/, firm_year_panel.csv, country_year.csv,
eurostat_comparison_lag0.csv, eurostat_comparison_lag1.csv and eurostat_summary.csv
(Table 5 in the software paper).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import aidisclose as aid
from aidisclose.validate import compare_with_external, eurostat_ai_share


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filings", required=True, help="CSV manually filtered to verified annual filings")
    ap.add_argument("--confirm-xbrl-periods", action="store_true", help="confirm the annual-period review")
    ap.add_argument("--min-firms", type=int, default=10)
    ap.add_argument("--out", default="case_esef")
    ap.add_argument("--eurostat-file", default=None, help="JSON-stat file; downloaded from the API if omitted")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if not a.confirm_xbrl_periods:
        ap.error("review and select annual filings, then specify --confirm-xbrl-periods")
    out = Path(a.out)

    src = aid.XbrlFilingsSource(cache_dir=out / "cache")
    filings = [
        aid.Filing.from_dict(row) for row in pd.read_csv(a.filings, dtype=str, keep_default_na=False).to_dict("records")
    ]
    if not filings or any(f.source != "xbrl" for f in filings):
        ap.error("filings must contain one or more reviewed XBRL records")
    print(f"{len(filings)} reviewed reports")
    pipe = aid.Pipeline(backend="rules", sections="core", recall_random=5, recall_probe_cap=10)
    results = pipe.run(filings, source=src, progress=True, checkpoint_dir=out / "checkpoint", resume=a.resume)
    results.save(out / "results")
    print(results.documents["status"].value_counts().to_string())

    panel = aid.firm_year_panel(results, xbrl_periods="reviewed")
    panel.to_csv(out / "firm_year_panel.csv", index=False)
    country = aid.group_panel(panel, by=["country", "fiscal_year"], min_firms=a.min_firms)
    country.to_csv(out / "country_year.csv", index=False)

    ext = eurostat_ai_share(path=a.eurostat_file, size_emp="GE250")
    rows = []
    for lag in (0, 1):
        merged, summary = compare_with_external(country, ext, year_lag=lag, min_firms=a.min_firms)
        merged.to_csv(out / f"eurostat_comparison_lag{lag}.csv", index=False)
        summary.insert(0, "year_lag", lag)
        rows.append(summary)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out / "eurostat_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
