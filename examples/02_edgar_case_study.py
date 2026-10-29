"""Case study 1: keyword versus substantive AI adoption in SEC annual reports.

Requires network access to sec.gov and matplotlib:

    pip install aidisclose matplotlib
    python examples/02_edgar_case_study.py --user-agent "Jane Doe jane.doe@university.edu"

Outputs (in ./case_edgar/): results/ (with a recall pool), qa.md,
firm_year_panel.csv, yearly.csv, state_mix.csv, filings.csv and
fig_prevalence.pdf/.png (Fig. 2 in the software paper).

The default ticker list is an illustrative cross-industry set of large
US-listed companies (Form 10-K); pass --tickers and --forms 10-K 20-F to add
foreign private issuers. The run is resumable: rerun with --resume after an
interruption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aidisclose as aid

DEFAULT_TICKERS = (
    "AAPL MSFT AMZN IBM INTC CSCO JPM BAC WFC GS JNJ PFE MRK UNH XOM CVX WMT HD KO PG "
    "CAT BA HON MMM UPS T VZ DIS MCD NKE"
).split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-agent", required=True, help="'Name email@domain' (required by the SEC)")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--forms", nargs="*", default=["10-K"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="case_edgar")
    a = ap.parse_args()
    out = Path(a.out)

    src = aid.EdgarSource(user_agent=a.user_agent, cache_dir=out / "cache")
    filings = src.list_filings(tickers=a.tickers, forms=a.forms, start_year=a.start, end_year=a.end)
    print(f"{len(filings)} filings ({', '.join(a.forms)})")
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame([f.to_dict() for f in filings]).to_csv(out / "filings.csv", index=False)  # reusable with --filings
    pipe = aid.Pipeline(backend="rules", sections="core", recall_random=5, recall_probe_cap=10)
    results = pipe.run(
        filings, source=src, progress=True, workers=a.workers, checkpoint_dir=out / "checkpoint", resume=a.resume
    )
    results.save(out / "results")
    from aidisclose.qa import quality_report, quality_to_markdown

    (out / "qa.md").write_text(quality_to_markdown(quality_report(results)), encoding="utf-8")

    panel = aid.firm_year_panel(results)
    panel.to_csv(out / "firm_year_panel.csv", index=False)
    yearly = aid.group_panel(panel, by=["fiscal_year"])
    yearly.to_csv(out / "yearly.csv", index=False)
    mix = aid.state_mix(results, by=["fiscal_year"])
    mix.to_csv(out / "state_mix.csv", index=False)
    print(
        yearly[
            [
                "fiscal_year",
                "n_firms",
                "share_keyword",
                "share_claim",
                "share_substantive",
                "gap_keyword_substantive_pp",
            ]
        ].to_string(index=False)
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for col, label, style in (
        ("share_keyword", "Keyword baseline", "-o"),
        ("share_claim", "Any focal-firm claim (WEAK or SUBSTANTIVE)", "-s"),
        ("share_substantive", "SUBSTANTIVE claim", "-^"),
    ):
        ax.plot(yearly["fiscal_year"], 100 * yearly[col], style, label=label, linewidth=1.5, markersize=4)
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Share of firms (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_prevalence.pdf")
    fig.savefig(out / "fig_prevalence.png", dpi=300)
    print(f"figure written to {out / 'fig_prevalence.pdf'}")


if __name__ == "__main__":
    main()
