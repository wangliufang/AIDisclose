"""Test helpers: fixture paths and an offline stand-in for requests.Session."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SRC = ROOT / "src"
if importlib.util.find_spec("aidisclose") is None:  # fallback only when no package is installed
    sys.path.insert(0, str(SRC))


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Routes URLs to fixture files; records every request."""

    def __init__(self, routes: dict[str, str | bytes | tuple[int, bytes]]):
        self.routes = routes
        self.calls: list[tuple[str, object]] = []
        self.posts: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        for prefix, target in self.routes.items():
            if url.startswith(prefix):
                if isinstance(target, tuple):
                    return FakeResponse(target[1], target[0])
                if isinstance(target, bytes):
                    return FakeResponse(target)
                return FakeResponse((FIXTURES / target).read_bytes())
        return FakeResponse(b"not found", 404)


def fixture(name: str) -> Path:
    return FIXTURES / name
