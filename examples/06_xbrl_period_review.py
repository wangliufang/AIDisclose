"""Demonstrate manual annual-period review on two real reports from one issuer.

Download the two source files identified in docs/xbrl_period_review.md and run:
python examples/06_xbrl_period_review.py --annual annual.xhtml --interim interim.xhtml
Inputs are checked against recorded SHA-256 digests and are not redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aidisclose import Filing, Pipeline, XbrlFilingsSource, firm_year_panel
from aidisclose.pipeline import Results

FILES = {
    "annual": ("2023-12-31", "bf17bb233b2d68166e41cecb1923ce45eabe868620966cb1afc9162b7e6ccdf3"),
    "interim": ("2023-06-30", "fa92a27aa07437dc4cd32b1c6cc1fe7e284cf09495b7ec065e8d650d47a3b6b1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual", required=True)
    parser.add_argument("--interim", required=True)
    a = parser.parse_args()
    source = XbrlFilingsSource(cache_dir=None)
    pipe = Pipeline(backend="rules")
    runs = {}
    for role, (period, digest) in FILES.items():
        data = Path(getattr(a, role)).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            parser.error(f"{role} input hash differs from the recorded report")
        filing = Filing(
            source="xbrl",
            filing_id=f"solar-{period}",
            firm_id="lei:21380031XTLI9X5MTY92",
            firm_name="SOLAR A/S",
            fiscal_year=2023,
            period_end=period,
            country="DK",
            form="ESEF",
            language="en",
        )
        document = source.make_document(filing, data, filename=role + ".xhtml")
        runs[role] = pipe.run_documents([document])
    combined = Results.concat([runs["annual"], runs["interim"]])
    try:
        firm_year_panel(combined, xbrl_periods="reviewed")
    except ValueError as exc:
        if "multiple XBRL period ends" not in str(exc):
            raise
    else:
        raise RuntimeError("annual/interim conflict was not rejected")
    panel = firm_year_panel(runs["annual"], xbrl_periods="reviewed")
    summary = {
        "issuer": "SOLAR A/S",
        "fiscal_year": 2023,
        "reports_processed": 2,
        "period_ends": [FILES["annual"][0], FILES["interim"][0]],
        "separate_run_document_status": {k: r.documents.iloc[0]["status"] for k, r in runs.items()},
        "candidate_counts": {k: len(r.sentences) for k, r in runs.items()},
        "mixed_period_firm_year_rejected": True,
        "reviewed_annual_panel_rows": len(panel),
        "limitations": "Two selected reports of one issuer; no human annotations or population inference.",
    }
    if any(v != "ok" for v in summary["separate_run_document_status"].values()) or len(panel) != 1:
        raise RuntimeError("report processing did not complete as expected")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
