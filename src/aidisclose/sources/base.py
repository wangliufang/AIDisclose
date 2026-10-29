"""Base classes for disclosure sources and a polite, caching HTTP client."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import requests

from ..models import Document, Filing
from ..text.extract import extract_text
from ..text.language import detect_language
from ..text.sections import split_sections

log = logging.getLogger(__name__)


class HttpClient:
    """HTTP GET with rate limiting, retries and an optional on-disk cache.

    Parameters
    ----------
    user_agent:
        Value of the ``User-Agent`` header. Some regulators (e.g. the SEC)
        require a declared agent with contact details.
    max_rps:
        Maximum requests per second sent to the network.
    cache_dir:
        Directory where raw responses are stored. Cached files are reused on
        later runs, which makes a study reproducible even if a source changes.
    """

    def __init__(
        self,
        user_agent: str,
        max_rps: float = 5.0,
        cache_dir: str | Path | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        session: requests.Session | None = None,
    ) -> None:
        if isinstance(max_rps, bool):
            raise ValueError("request rate must be finite and greater than zero")
        try:
            rate = float(max_rps)
        except (TypeError, ValueError) as exc:
            raise ValueError("request rate must be finite and greater than zero") from exc
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(1.0 / rate):
            raise ValueError("request rate must be finite and greater than zero")
        self.user_agent = user_agent
        self.min_interval = 1.0 / rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._lock = threading.Lock()
        self._last = 0.0
        self._shared_rate = None
        self.n_requests = 0

    # The lock and the session cannot be pickled; worker processes get fresh ones.
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_lock", None)
        session = state.pop("session", None)
        state["_custom_session"] = session if not isinstance(session, requests.Session) else None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        custom = state.pop("_custom_session", None)
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self.session = custom if custom is not None else requests.Session()

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = url + ("?" + json.dumps(params, sort_keys=True) if params else "")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / digest

    def _throttle(self) -> None:
        if self._shared_rate is not None:
            lock, last, interval = self._shared_rate
            with lock:
                wait = interval - (time.monotonic() - last.value)
                if wait > 0:
                    time.sleep(wait)
                last.value = time.monotonic()
            return
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get_bytes(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> bytes:
        path = self._cache_path(url, params)
        if use_cache and path is not None and path.exists():
            return path.read_bytes()
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                self.n_requests += 1
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {url}", response=resp)
                resp.raise_for_status()
                data = resp.content
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Readers must never see a partially written cache entry.
                    tmp = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as f:
                            tmp = Path(f.name)
                            f.write(data)
                        os.replace(tmp, path)
                    finally:
                        if tmp is not None:
                            tmp.unlink(missing_ok=True)
                return data
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
            except requests.RequestException as exc:
                last_exc = exc
            if attempt < self.max_retries:
                delay = min(2**attempt, 60)
                log.info("retrying %s in %ss (%s)", url, delay, last_exc)
                time.sleep(delay)
        raise RuntimeError(f"GET {url} failed after {self.max_retries + 1} attempts: {last_exc}")

    def get_json(self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        return json.loads(self.get_bytes(url, params=params, use_cache=use_cache).decode("utf-8"))


class Source(ABC):
    """A provider of disclosure documents."""

    name: str = "source"

    def config(self) -> dict[str, Any]:
        """Return the non-secret source settings used for provenance."""
        return {"name": self.name}

    @abstractmethod
    def list_filings(self, **query: Any) -> list[Filing]:
        """Return metadata for the filings matching *query*."""

    @abstractmethod
    def fetch_document(self, filing: Filing) -> Document:
        """Download (or read) a filing and return its extracted text."""

    def iter_documents(self, filings: Iterable[Filing]):
        for f in filings:
            yield self.fetch_document(f)

    @staticmethod
    def make_document(filing: Filing, data: bytes, filename: str = "", sections_fn="auto") -> Document:
        """Extract text from *data* and split it into sections.

        ``sections_fn="auto"`` (default) uses the Item schema of the filing's
        form (10-K, 20-F) or, for other documents, the annual-report part
        splitter; ``None`` keeps the document whole; a callable
        ``text -> {label: text}`` is used as given.
        """
        text = extract_text(data, filename=filename)
        if sections_fn == "auto":
            sections = split_sections(text, filing.form)
        else:
            sections = sections_fn(text) if sections_fn else {}
        if list(sections) == ["full"]:
            sections = {}
        lang, scores = detect_language(text)
        return Document(filing=filing, text=text, sections=sections, language=lang, language_scores=scores)


def fiscal_year_from_period_end(period_end: str | None, convention: str = "jan_may_previous") -> int | None:
    """Map a period end to a fiscal-year label using an explicit convention.

    ``period_end_year`` labels a period with its end year. The backwards-
    compatible ``jan_may_previous`` convention labels January-May periods
    with the previous year. The selected convention should match the panel
    definition used by the study.
    """
    if not period_end:
        return None
    if convention not in {"period_end_year", "jan_may_previous"}:
        raise ValueError("convention must be 'period_end_year' or 'jan_may_previous'")
    try:
        if len(period_end) != 10 or period_end[4] != "-" or period_end[7] != "-":
            return None
        parsed = date.fromisoformat(period_end)
        year, month = parsed.year, parsed.month
    except (ValueError, TypeError, IndexError):
        return None
    return year if convention == "period_end_year" or month >= 6 else year - 1
