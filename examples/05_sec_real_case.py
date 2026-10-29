"""Run and prepare a blind annotation sample from one public SEC filing.

Download the 2024 SES AI 10-K from the SEC link in docs/real_case.md, then:
  python examples/05_sec_real_case.py --input /path/to/ses.htm --out /new/output
The report is fetched separately; only an aggregate run summary is distributed.
Never interpret the predictions as human-validated accuracy or firm prevalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "0719fda6d04e08b041396f193243f1149881ee20ad2ab779fc37e7c8f9dd1861"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="downloaded SES AI 2024 10-K HTML file")
    p.add_argument("--out", required=True, help="new or empty output directory")
    a = p.parse_args()
    raw = Path(a.input).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        p.error(f"file bytes do not match the archived case: {digest}; expected {EXPECTED_SHA256}")
    out = Path(a.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        p.error("output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        (source / "ses_2024_10k.htm").write_bytes(raw)
        (source / "metadata.csv").write_text(
            "file,firm_id,firm_name,fiscal_year,country,form,language\n"
            "ses_2024_10k.htm,SES,SES AI Corporation,2024,US,10-K,en\n",
            encoding="utf-8",
        )

        def run(*args: str) -> None:
            subprocess.run([sys.executable, "-m", "aidisclose.cli", *map(str, args)], check=True)

        run(
            "run",
            "--source",
            "local",
            "--dir",
            source,
            "--metadata",
            source / "metadata.csv",
            "--out",
            out / "results",
            "--recall-random",
            "3",
            "--recall-probe-cap",
            "3",
            "--quiet",
        )
        run(
            "sample",
            "--results",
            out / "results",
            "--n",
            "60",
            "--min-per-stratum",
            "5",
            "--seed",
            "2026",
            "--out",
            out / "sample.csv",
        )
        shutil.copyfile(out / "sample.csv", out / "annotator_A.csv")
        shutil.copyfile(out / "sample.csv", out / "annotator_B.csv")
        doc = pd.read_csv(out / "results" / "documents.csv", keep_default_na=False).iloc[0]
        sentences = pd.read_csv(out / "results" / "sentences.csv", keep_default_na=False)
        summary = {
            "input_sha256": digest,
            "report": "SES AI Corporation 2024 Form 10-K",
            "status": doc["status"],
            "document_words": int(doc["n_words"]),
            "processed_sentences": int(doc["n_sentences"]),
            "candidate_sentences": len(sentences),
            "predicted_states": {k: int(v) for k, v in sentences["state"].value_counts().items()},
            "classifier_errors": int(doc["n_classifier_errors"]),
            "sample_n": len(pd.read_csv(out / "sample.csv")),
            "interpretation": "One selected filing; machine predictions only, no human validation or prevalence inference.",
        }
        (out / "case_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
