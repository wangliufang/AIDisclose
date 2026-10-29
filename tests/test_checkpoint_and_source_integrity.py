"""Regressions for checkpoint identity, metadata, caching and validation."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import FakeResponse, FakeSession
from test_backends import FakeChat, cand

import aidisclose
from aidisclose import Document, Filing, LocalSource, Pipeline
from aidisclose.classify.llm import LLMClassifier, parse_response
from aidisclose.pipeline import Checkpoint, DocumentOutcome
from aidisclose.sources.base import HttpClient, Source


class RecordingSession:
    def __init__(self, path):
        self.path = path

    def get(self, url, **kwargs):
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(str(time.monotonic()) + "\n")
        return FakeResponse(b"fixture")


class RecordingSource(Source):
    name = "test"

    def __init__(self, path):
        self.http = HttpClient("test@example.org", max_rps=10, session=RecordingSession(path))

    def list_filings(self, **kwargs):
        return [
            Filing(source="test", filing_id=str(i), firm_id=str(i), firm_name="Acme", fiscal_year=2024)
            for i in range(4)
        ]

    def fetch_document(self, filing):
        self.http.get_bytes("https://fixture.invalid/" + filing.filing_id)
        return Document(filing=filing, language="en", text="We use machine learning to forecast demand in our stores.")


class TestFeatureIntegrity(unittest.TestCase):
    def test_wheel_suite_uses_installed_package_when_requested(self):
        root = os.environ.get("AIDISCLOSE_EXPECT_INSTALL_ROOT")
        if root:
            self.assertTrue(
                Path(aidisclose.__file__).resolve().is_relative_to(Path(root).resolve()), aidisclose.__file__
            )

    def test_llm_invalid_reason_is_not_fabricated_or_cached(self):
        with self.assertRaisesRegex(ValueError, "valid reason"):
            parse_response('{"state":"NO_CLAIM","reason":"DEPLOYMENT"}')
        with self.assertRaisesRegex(ValueError, "valid reason"):
            parse_response('{"state":"WEAK"}')
        chat = FakeChat(['{"state":"NO_CLAIM","reason":"DEPLOYMENT"}', '{"state":"NO_CLAIM","reason":"THIRD_PARTY"}'])
        model = LLMClassifier(model="fake", session=chat, cache_path=None, max_retries=1)
        with mock.patch("aidisclose.classify.llm.time.sleep"):
            pred = model.classify_one(cand("Our competitors use AI."))
        self.assertEqual(pred.reason.value, "THIRD_PARTY")
        self.assertEqual(len(chat.bodies), 2)

    def test_conflicting_basename_and_duplicate_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a").mkdir()
            (d / "b").mkdir()
            (d / "a/report.html").write_text("a")
            (d / "b/report.html").write_text("b")
            meta = d / "meta.csv"
            meta.write_text("file,firm_id\nreport.html,A\n")
            with self.assertRaisesRegex(ValueError, "ambiguous metadata basename"):
                LocalSource(d, meta).list_filings()
            meta.write_text("file,firm_id\na/report.html,A\na/report.html,B\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                LocalSource(d, meta).list_filings()
            meta.write_text("file,firm_id\na/report.html,A\nb/report.html,B\n")
            self.assertEqual([x.firm_id for x in LocalSource(d, meta).list_filings()], ["A", "B"])

    def test_explicit_local_language_is_honoured(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "short.txt").write_text("We use machine learning to forecast customer demand.")
            meta = d / "meta.csv"
            meta.write_text("file,firm_id,language\nshort.txt,A,EN\n")
            source = LocalSource(d, meta)
            filing = source.list_filings()[0]
            self.assertEqual(source.fetch_document(filing).language, "en")
            self.assertEqual(Pipeline().run([filing], source).documents.status.iloc[0], "ok")
            meta.write_text("file,firm_id,language\nshort.txt,A,de\n")
            source = LocalSource(d, meta)
            filing = source.list_filings()[0]
            self.assertEqual(Pipeline().run([filing], source).documents.status.iloc[0], "skipped_non_english")

    def test_mismatched_checkpoint_parts_are_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            c = Checkpoint(td, "x")
            c.write("a", DocumentOutcome([], {"filing_id": "a", "status": "ok"}))
            c.write("b", DocumentOutcome([], {"filing_id": "b", "status": "ok"}))
            a, b = c._path("a"), c._path("b")
            contents = a.read_bytes()
            a.write_bytes(b.read_bytes())
            b.write_bytes(contents)
            c = Checkpoint(td, "x", resume=True)
            self.assertEqual(c.statuses(), {})
            self.assertIsNone(c.read("a"))
            (Path(td) / "checkpoint.json").unlink()
            with self.assertRaisesRegex(ValueError, "metadata is missing"):
                Checkpoint(td, "x", resume=True)
            self.assertTrue(a.exists() and b.exists())

    def test_cache_replace_is_atomic_and_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            h = HttpClient(
                "test@example.org", max_rps=1000, cache_dir=td, session=FakeSession({"https://x": b"complete"})
            )
            path = h._cache_path("https://x", None)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old-complete")
            real_replace = os.replace

            def inspect(src, dst):
                self.assertEqual(Path(dst).read_bytes(), b"old-complete")
                self.assertEqual(Path(src).read_bytes(), b"complete")
                real_replace(src, dst)

            with mock.patch("aidisclose.sources.base.os.replace", side_effect=inspect):
                self.assertEqual(h.get_bytes("https://x", use_cache=False), b"complete")
            self.assertEqual(path.read_bytes(), b"complete")
            with mock.patch("aidisclose.sources.base.os.replace", side_effect=OSError("fail")):
                with self.assertRaises(OSError):
                    h.get_bytes("https://x", use_cache=False)
            self.assertEqual(path.read_bytes(), b"complete")

    def test_process_requests_share_one_rate_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "times.txt")
            source = RecordingSource(path)
            result = Pipeline().run(source.list_filings(), source, workers=4, executor="process")
            self.assertEqual(len(result.documents), 4)
            self.assertTrue(result.documents.status.eq("ok").all())
            ordered = sorted(float(t) for t in Path(path).read_text().splitlines())
            self.assertEqual(len(ordered), 4)
            self.assertTrue(all(b - a >= 0.07 for a, b in zip(ordered, ordered[1:])), ordered)
