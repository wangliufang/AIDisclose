import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from helpers import FakeResponse, fixture

from aidisclose.classify import get_classifier
from aidisclose.classify.llm import PROMPT_SHA256, LLMClassifier, parse_response
from aidisclose.lexicon import Lexicon
from aidisclose.models import AdoptionState, Candidate, Reason

LEX = Lexicon.load("default", tiers=None)


def cand(text, firm="Fictional Corp.", section="10-K:1"):
    h = LEX.find(text)
    return Candidate(
        filing_id="f",
        sentence_id=text[:20],
        section=section,
        text=text,
        firm_name=firm,
        terms=h.term_ids("valid"),
        term_spans=[(x.start, x.end) for x in h.valid],
    )


class FakeChat:
    """Stand-in for an OpenAI-compatible server."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append({"url": url, "json": json, "headers": headers})
        content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        payload = {"choices": [{"message": {"content": content}}]}
        import json as _json

        return FakeResponse(_json.dumps(payload).encode())


class LLMTests(unittest.TestCase):
    def test_parse_response(self):
        st, rs, ev = parse_response('Sure! {"state": "substantive", "reason": "DEPLOYMENT", "evidence": "we use"}')
        self.assertEqual((st, rs, ev), (AdoptionState.SUBSTANTIVE, Reason.DEPLOYMENT, "we use"))
        with self.assertRaises(ValueError):
            parse_response('{"state": "NO_CLAIM", "reason": "DEPLOYMENT"}')
        st, rs, _ = parse_response('{"state": "A2", "reason": "VAGUE"}')
        self.assertEqual(st, AdoptionState.WEAK)
        with self.assertRaises(ValueError):
            parse_response("no json here")

    def test_request_cache_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = FakeChat(['{"state": "SUBSTANTIVE", "reason": "PRODUCT", "evidence": "launched"}'])
            clf = LLMClassifier(
                model="test-model",
                base_url="http://localhost:9/v1",
                api_key="SECRET-KEY-123",
                cache_path=Path(tmp) / "c.sqlite",
                session=chat,
            )
            c = cand("In 2024 we launched an AI assistant for customers.")
            p = clf.classify([c])[0]
            self.assertEqual((p.state, p.reason), (AdoptionState.SUBSTANTIVE, Reason.PRODUCT))
            body = chat.bodies[0]
            self.assertTrue(body["url"].endswith("/chat/completions"))
            self.assertEqual(body["json"]["temperature"], 0.0)
            self.assertEqual(body["json"]["response_format"]["type"], "json_schema")
            self.assertEqual(body["headers"]["Authorization"], "Bearer SECRET-KEY-123")
            self.assertIn("<target>In 2024 we launched", body["json"]["messages"][-1]["content"])
            clf.classify([c])
            self.assertEqual(len(chat.bodies), 1)  # second call answered from the cache
            cfg = clf.config()
            self.assertEqual(cfg["prompt_sha256"], PROMPT_SHA256)
            self.assertEqual(cfg["endpoint"], "http://localhost:9/v1")
            self.assertNotIn("SECRET-KEY-123", json.dumps(cfg))  # credentials never enter provenance

    def test_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = FakeChat(
                [
                    '{"state": "WEAK", "reason": "VAGUE"}',
                    '{"state": "SUBSTANTIVE", "reason": "DEPLOYMENT"}',
                    '{"state": "WEAK", "reason": "HEDGED"}',
                ]
            )
            clf = LLMClassifier(model="m", temperature=0.7, n_votes=3, cache_path=Path(tmp) / "c.sqlite", session=chat)
            p = clf.classify_one(cand("We use AI across our business."))
            self.assertEqual(p.state, AdoptionState.WEAK)
            self.assertAlmostEqual(p.details["agreement"], 2 / 3, places=3)
            self.assertEqual([b["json"]["seed"] for b in chat.bodies], [0, 1, 2])

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            LLMClassifier(model="m", n_votes=3, temperature=0.0, cache_path=None)
        with mock.patch.dict(os.environ, {"AIDISCLOSE_LLM_MODEL": ""}):
            with self.assertRaises(ValueError):
                LLMClassifier(model=None, cache_path=None)

    def test_unparseable_reply_is_flagged(self):
        clf = LLMClassifier(model="m", cache_path=None, session=FakeChat(["I cannot answer that."]))
        p = clf.classify_one(cand("We use AI."))
        self.assertEqual(p.confidence, 0.0)
        self.assertIn("error", p.details)


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sklearn  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("scikit-learn not installed") from None
        from importlib import resources

        cases = yaml.safe_load(fixture("rule_cases.yaml").read_text(encoding="utf-8"))
        demo = pd.read_csv(Path(str(resources.files("aidisclose.demo").joinpath("data"))) / "demo_gold.csv")
        cls.cands = [cand(c["text"], section=c.get("section", "10-K:1")) for c in cases] + [
            cand(t) for t in demo["text"]
        ]
        cls.labels = [c["state"] for c in cases] + demo["gold_state"].tolist()

    def test_train_predict_save_load(self):
        from aidisclose.classify.supervised import SupervisedClassifier

        model = SupervisedClassifier.train(self.cands, self.labels)
        preds = model.classify(self.cands[:5])
        self.assertEqual(len(preds), 5)
        self.assertTrue(all(0 <= p.confidence <= 1 for p in preds))
        self.assertIn("probabilities", preds[0].details)
        with tempfile.TemporaryDirectory() as tmp:
            path = model.save(Path(tmp) / "m.joblib")
            again = get_classifier("supervised", model_path=path)
            self.assertEqual([p.state for p in again.classify(self.cands[:5])], [p.state for p in preds])
        self.assertEqual(model.config()["n_train"], len(self.labels))

    def test_cross_validate(self):
        from aidisclose.classify.supervised import SupervisedClassifier

        rep = SupervisedClassifier.cross_validate(self.cands, self.labels, k=3)
        self.assertEqual(rep["n"], len(self.labels))
        self.assertEqual(rep["k"], 3)

    def test_requires_two_classes(self):
        from aidisclose.classify.supervised import SupervisedClassifier

        with self.assertRaises(ValueError):
            SupervisedClassifier.train(self.cands[:3], ["WEAK"] * 3)


class RegistryTests(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            get_classifier("magic")
        with self.assertRaises(ValueError):
            get_classifier("supervised")


if __name__ == "__main__":
    unittest.main()
