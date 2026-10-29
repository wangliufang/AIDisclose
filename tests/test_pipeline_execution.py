"""Pipeline checkpoints, workers, recall pool and language routing."""

import json
import pickle
import tempfile
import unittest
from importlib import resources
from pathlib import Path

import pandas as pd
from helpers import FakeSession

from aidisclose import LocalSource, Pipeline, Results
from aidisclose.lexicon import Lexicon
from aidisclose.models import Document, Filing
from aidisclose.sources.base import HttpClient


def demo_dir() -> Path:
    return Path(str(resources.files("aidisclose.demo").joinpath("data")))


def demo():
    d = demo_dir()
    src = LocalSource(d, metadata=d / "metadata.csv")
    return src, src.list_filings()


def comparable(res: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = res.sentences.sort_values("sentence_id").reset_index(drop=True)
    d = res.documents.sort_values("filing_id").reset_index(drop=True)
    return s[["sentence_id", "state", "reason", "confidence"]], d.drop(columns=["extra"])


class CountingSource:
    """Wraps a source and fails selected filings once (to test resume)."""

    name = "counting"

    def __init__(self, inner, fail_ids=()):
        self.inner = inner
        self.fail_ids = set(fail_ids)
        self.fetched: list[str] = []

    def fetch_document(self, filing):
        self.fetched.append(filing.filing_id)
        if filing.filing_id in self.fail_ids:
            self.fail_ids.discard(filing.filing_id)
            raise ConnectionError("transient failure")
        return self.inner.fetch_document(filing)


class CheckpointTests(unittest.TestCase):
    def test_resume_skips_done_and_retries_errors(self):
        src, filings = demo()
        reference = Pipeline().run(filings, source=src)
        with tempfile.TemporaryDirectory() as tmp:
            first_fail = filings[3].filing_id
            s1 = CountingSource(src, fail_ids=[first_fail])
            r1 = Pipeline().run(filings, source=s1, checkpoint_dir=tmp)
            self.assertEqual(r1.documents.set_index("filing_id").loc[first_fail, "status"], "error")
            self.assertEqual(len(list(Path(tmp, "parts").glob("*.json"))), len(filings))
            s2 = CountingSource(src)
            r2 = Pipeline().run(filings, source=s2, checkpoint_dir=tmp, resume=True)
            self.assertEqual(s2.fetched, [first_fail])  # only the failed filing is fetched again
            a, b = comparable(r2), comparable(reference)
            pd.testing.assert_frame_equal(a[0], b[0])
            pd.testing.assert_frame_equal(a[1], b[1], check_dtype=False)
            self.assertEqual(list(r2.documents["filing_id"]), [f.filing_id for f in filings])  # input order

    def test_resume_refuses_other_configuration(self):
        src, filings = demo()
        with tempfile.TemporaryDirectory() as tmp:
            Pipeline().run(filings[:2], source=src, checkpoint_dir=tmp)
            with self.assertRaises(ValueError):
                Pipeline(tiers=None).run(filings[:2], source=src, checkpoint_dir=tmp, resume=True)
            # without resume the old checkpoints are discarded
            s = CountingSource(src)
            Pipeline(tiers=None).run(filings[:2], source=s, checkpoint_dir=tmp)
            self.assertEqual(len(s.fetched), 2)

    def test_partial_checkpoint_file_is_ignored(self):
        src, filings = demo()
        with tempfile.TemporaryDirectory() as tmp:
            Pipeline().run(filings[:2], source=CountingSource(src), checkpoint_dir=tmp)
            part = sorted(Path(tmp, "parts").glob("*.json"))[0]
            part.write_text('{"filing_id": "trunc', encoding="utf-8")  # crash during a write
            s = CountingSource(src)
            res = Pipeline().run(filings[:2], source=s, checkpoint_dir=tmp, resume=True)
            self.assertEqual(len(s.fetched), 1)
            self.assertEqual(len(res.documents), 2)

    def test_fingerprint_stable(self):
        self.assertEqual(Pipeline().fingerprint(), Pipeline().fingerprint())
        self.assertNotEqual(Pipeline().fingerprint(), Pipeline(sections="core").fingerprint())


class WorkerTests(unittest.TestCase):
    def test_threads_and_processes_match_sequential(self):
        src, filings = demo()
        ref = comparable(Pipeline().run(filings, source=src))
        for executor in ("thread", "process"):
            res = Pipeline().run(filings, source=src, workers=3, executor=executor)
            got = comparable(res)
            pd.testing.assert_frame_equal(got[0], ref[0])
            pd.testing.assert_frame_equal(got[1], ref[1], check_dtype=False)
            self.assertEqual(list(res.documents["filing_id"]), [f.filing_id for f in filings])
        with self.assertRaises(ValueError):
            Pipeline().run(filings, source=src, workers=2, executor="gpu")

    def test_http_client_pickles(self):
        http = HttpClient("UA test@example.org", max_rps=4, cache_dir=None)
        clone = pickle.loads(pickle.dumps(http))
        self.assertEqual(clone.min_interval, http.min_interval)
        self.assertIsNot(clone._lock, http._lock)
        fake = HttpClient("UA test@example.org", session=FakeSession({"https://x.org": b"ok"}))
        self.assertEqual(pickle.loads(pickle.dumps(fake)).get_bytes("https://x.org/a"), b"ok")

    def test_rate_limit_shared_between_processes(self):
        from aidisclose.pipeline import _http_clients, _init_worker

        class Src:
            def __init__(self):
                self.http = HttpClient("UA test@example.org", max_rps=8, cache_dir=None)

        s = Src()
        self.assertEqual(len(_http_clients(s)), 1)
        shared = (object(), object(), 1 / 8)
        _init_worker(Pipeline(), s, 4, shared)
        self.assertIs(s.http._shared_rate, shared)
        self.assertAlmostEqual(s.http.min_interval, 1 / 8)


class RecallPoolTests(unittest.TestCase):
    def test_pool(self):
        src, filings = demo()
        pipe = Pipeline(recall_random=2, recall_probe_cap=3)
        res = pipe.run(filings, source=src)
        pool = res.recall_pool
        self.assertGreater(len(pool), 0)
        self.assertTrue(set(pool["pool"]) <= {"probe", "random"})
        # pool sentences were not retrieved by either lexicon
        self.assertFalse(set(pool["sentence_id"]) & set(res.sentences["sentence_id"]))
        docs = res.documents.set_index("filing_id")
        for (fid, name), grp in pool.groupby(["filing_id", "pool"]):
            size = docs.loc[fid, "n_probe_sentences" if name == "probe" else "n_other_sentences"]
            self.assertTrue((grp["stratum_size"] == size).all())
            self.assertEqual(len(grp), grp["n_drawn"].iloc[0])
            self.assertLessEqual(len(grp), 3 if name == "probe" else 2)
            self.assertAlmostEqual(grp["inclusion_prob"].iloc[0], len(grp) / size, places=6)
        probe = pool[pool["pool"] == "probe"]
        self.assertTrue((probe["probe_terms"] != "").all())
        self.assertIn("recall_pool", res.provenance)
        # reproducible, saved and loaded
        again = Pipeline(recall_random=2, recall_probe_cap=3).run(filings, source=src)
        self.assertEqual(list(again.recall_pool["sentence_id"]), list(pool["sentence_id"]))
        with tempfile.TemporaryDirectory() as tmp:
            res.save(tmp)
            loaded = Results.load(tmp)
            self.assertEqual(len(loaded.recall_pool), len(pool))
            self.assertEqual(loaded.recall_pool["stratum_size"].dtype.kind, "i")
        # no pool by default, and old results without the file still load
        self.assertEqual(len(Pipeline().run(filings[:2], source=src).recall_pool), 0)

    def test_probe_lexicon(self):
        probe = Lexicon.load("recall-probe", tiers=None)
        self.assertEqual(probe.find("Our routing algorithms cut fuel use.").term_ids("raw"), ["algorithm"])
        self.assertEqual(probe.find("Revenue grew 4%.").term_ids("raw"), [])


class LanguageRoutingTests(unittest.TestCase):
    DE_LEXICON = """name: test-de
version: "0.0.1"
language: de
terms:
  - id: ki
    pattern: 'Künstliche[nr]?\\s+Intelligenz|KI'
    case_sensitive: true
    category: general
    tier: core
"""

    def doc(self, text, lang):
        f = Filing(source="t", filing_id=f"t:{lang}", firm_id=lang, firm_name="Muster AG", fiscal_year=2024)
        return Document(filing=f, text=text, language=lang)

    def test_routing(self):
        from aidisclose.classify.base import Classifier
        from aidisclose.models import AdoptionState, Prediction, Reason

        class Always(Classifier):
            name = "always-weak"

            def classify(self, candidates):
                return [Prediction(AdoptionState.WEAK, Reason.VAGUE, 0.5, self.name, {}) for _ in candidates]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "de.yaml")
            path.write_text(self.DE_LEXICON, encoding="utf-8")
            pipe = Pipeline(backend=Always(), languages=("en", "de"), lexicons={"de": path})
            res = pipe.run_documents(
                [
                    self.doc("Wir setzen Künstliche Intelligenz in der Logistik ein.", "de"),
                    self.doc("We use artificial intelligence in logistics.", "en"),
                    self.doc("Nous utilisons l'intelligence artificielle.", "fr"),
                ]
            )
            status = dict(zip(res.documents["language"], res.documents["status"]))
            self.assertEqual(status, {"de": "ok", "en": "ok", "fr": "skipped_language"})
            de = res.sentences[res.sentences["firm_id"] == "de"]
            self.assertEqual(de["terms"].tolist(), ["ki"])
            self.assertEqual(de["baseline_terms"].tolist(), ["ki"])  # baseline defaults to the language lexicon
            self.assertIn("lexicons", res.provenance)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Pipeline(languages=("en", "de"))  # no German lexicon shipped
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "de.yaml")
            path.write_text(self.DE_LEXICON, encoding="utf-8")
            with self.assertRaises(ValueError):
                Pipeline(languages=("de",), lexicons={"de": path})  # rules backend is English-only
        res = Pipeline().run_documents([self.doc("Wir setzen KI ein und mehr Text.", "de")])
        self.assertEqual(res.documents["status"].tolist(), ["skipped_non_english"])
        all_lang = Pipeline(english_only=False)
        self.assertIsNone(all_lang.languages)


class CheckpointJsonTests(unittest.TestCase):
    def test_numpy_values_serialise(self):
        import numpy as np

        from aidisclose.pipeline import _json_default

        self.assertEqual(json.loads(json.dumps({"a": np.int64(3)}, default=_json_default)), {"a": 3})
        self.assertIsNone(_json_default(pd.NA))


if __name__ == "__main__":
    unittest.main()
