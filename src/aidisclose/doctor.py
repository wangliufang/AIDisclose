"""Environment checks (``aidisclose doctor``).

Reports the Python and package versions, optional dependencies, whether the
cache directory is writable and, with ``network=True``, whether the public
data sources (and an LLM endpoint, if configured) can be reached. Network
checks send one small request each and identify themselves with the given
User-Agent.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from ._version import __version__

PACKAGES = [
    ("pandas", True, "core"),
    ("numpy", True, "core"),
    ("requests", True, "core"),
    ("lxml", True, "core"),
    ("bs4", True, "core (fallback HTML parser)"),
    ("yaml", True, "core"),
    ("openpyxl", True, "Excel annotation sheets"),
    ("pypdf", False, "PDF reports (aidisclose[pdf])"),
    ("sklearn", False, "supervised backend (aidisclose[ml])"),
    ("transformers", False, "experimental transformer backend"),
    ("torch", False, "experimental transformer backend"),
    ("matplotlib", False, "example figures"),
]
_DIST = {"bs4": "beautifulsoup4", "yaml": "PyYAML", "sklearn": "scikit-learn"}
ENDPOINTS = {
    "SEC EDGAR (submissions API)": "https://data.sec.gov/submissions/CIK0000320193.json",
    "filings.xbrl.org": "https://filings.xbrl.org/api/filings?page%5Bsize%5D=1",
    "Eurostat dissemination API": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "isoc_eb_ai?format=JSON&lang=EN&geo=DK&time=2023&size_emp=GE250&indic_is=E_AI_TANY&unit=PC_ENT",
}


def _version(mod: str) -> str | None:
    try:
        return md.version(_DIST.get(mod, mod))
    except md.PackageNotFoundError:
        try:
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "installed")
        except ImportError:
            return None


def check_environment(cache_dir: str | Path = ".aidisclose_cache") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"check": "aidisclose", "status": "ok", "detail": __version__},
        {
            "check": "python",
            "status": "ok" if sys.version_info >= (3, 10) else "fail",
            "detail": f"{platform.python_version()} ({platform.platform()})",
        },
    ]
    for mod, required, purpose in PACKAGES:
        v = _version(mod)
        if v is None:
            rows.append(
                {"check": mod, "status": "fail" if required else "missing", "detail": f"not installed: {purpose}"}
            )
        else:
            rows.append({"check": mod, "status": "ok", "detail": f"{v} ({purpose})"})
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=True):
            pass
        rows.append({"check": "cache directory", "status": "ok", "detail": str(Path(cache_dir).resolve())})
    except OSError as exc:
        rows.append({"check": "cache directory", "status": "fail", "detail": f"{cache_dir}: {exc}"})
    return rows


def check_network(user_agent: str, llm_base_url: str | None = None, timeout: float = 15.0, session=None):
    import requests

    sess = session or requests.Session()
    rows = []
    targets = dict(ENDPOINTS)
    if llm_base_url:
        targets["LLM endpoint (/models)"] = llm_base_url.rstrip("/") + "/models"
    for name, url in targets.items():
        headers = {"User-Agent": user_agent}
        if name.startswith("LLM") and os.environ.get("AIDISCLOSE_LLM_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['AIDISCLOSE_LLM_API_KEY']}"
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            ok = resp.status_code < 400
            rows.append({"check": name, "status": "ok" if ok else "fail", "detail": f"HTTP {resp.status_code}"})
        except requests.RequestException as exc:
            rows.append({"check": name, "status": "fail", "detail": f"{type(exc).__name__}: {str(exc)[:120]}"})
    return rows


def format_rows(rows: list[dict[str, Any]]) -> str:
    width = max(len(r["check"]) for r in rows)
    mark = {"ok": "ok     ", "missing": "missing", "fail": "FAIL   "}
    return "\n".join(f"{r['check']:<{width}}  {mark.get(r['status'], r['status'])}  {r['detail']}" for r in rows)
