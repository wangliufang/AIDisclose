"""Record real API responses as offline fixtures (needs network access).

The unit tests use small hand-made fixtures so that they run offline. This
script downloads current responses from SEC EDGAR, filings.xbrl.org and
Eurostat into a separate folder, with a manifest of URLs, retrieval times and
SHA-256 digests, so that adapters can be re-checked against real data and
fixtures refreshed when a source changes its format.

    python scripts/record_fixtures.py --user-agent "Jane Doe jane@uni.edu" --out recorded/

Nothing in tests/fixtures is overwritten. Respect the sources' terms of use:
the script sends a handful of requests, well below the SEC's 10 requests per
second.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from aidisclose.sources.base import HttpClient
from aidisclose.sources.edgar import INDEX_URL, SUBMISSIONS_URL, parse_filing_index

EUROSTAT = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{ds}"
    "?format=JSON&lang=EN&indic_is=E_AI_TANY&unit=PC_ENT{extra}"
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user-agent", required=True, help="'Name email@domain' (required by the SEC)")
    p.add_argument("--out", default="recorded")
    p.add_argument("--cik", type=int, nargs="*", default=[320193, 1000184, 16868], help="CIKs (default AAPL, SAP, CNI)")
    p.add_argument("--forms", nargs="*", default=["10-K", "20-F", "40-F"])
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    http = HttpClient(args.user_agent, max_rps=4, cache_dir=None)
    manifest = []

    def save(name: str, url: str) -> bytes:
        data = http.get_bytes(url, use_cache=False)
        (out / name).write_bytes(data)
        manifest.append(
            {
                "file": name,
                "url": url,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
        print(f"{name}: {len(data):,} bytes")
        return data

    for cik in args.cik:
        sub = json.loads(save(f"edgar_submissions_{cik}.json", SUBMISSIONS_URL.format(cik=cik)))
        recent = sub.get("filings", {}).get("recent", {})
        for i, form in enumerate(recent.get("form", [])):
            if form in args.forms:
                acc = recent["accessionNumber"][i]
                idx = save(
                    f"edgar_index_{cik}_{acc}.htm", INDEX_URL.format(cik=cik, acc_nodash=acc.replace("-", ""), acc=acc)
                )
                rows = parse_filing_index(idx)
                (out / f"edgar_index_{cik}_{acc}.parsed.json").write_text(json.dumps(rows, indent=1))
                break
    save(
        "xbrl_filings_dk_page1.json",
        "https://filings.xbrl.org/api/filings?include=entity&page%5Bsize%5D=20&filter="
        "%5B%7B%22name%22%3A%22country%22%2C%22op%22%3A%22eq%22%2C%22val%22%3A%22DK%22%7D%5D",
    )
    save("eurostat_isoc_eb_ai.json", EUROSTAT.format(ds="isoc_eb_ai", extra="&size_emp=GE250"))
    save("eurostat_isoc_eb_ain2.json", EUROSTAT.format(ds="isoc_eb_ain2", extra="&size_emp=GE10"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest)} responses recorded in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
