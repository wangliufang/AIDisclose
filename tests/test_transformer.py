"""The experimental transformer backend, exercised with stand-ins for torch/transformers."""

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from aidisclose.models import AdoptionState, Candidate, Reason


class T:
    """Minimal tensor stand-in."""

    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def softmax(self, dim):
        out = []
        for row in self.data:
            m = max(row)
            e = [math.exp(x - m) for x in row]
            out.append([x / sum(e) for x in e])
        return T(out)

    def tolist(self):
        return self.data


class FakeTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def __call__(self, first, second, **kw):
        FakeTokenizer.calls.append((list(first), list(second), kw))
        return {"input_ids": T([[len(x)] for x in first])}

    def save_pretrained(self, out):
        Path(out, "tokenizer.json").write_text("{}")


class FakeModel:
    def __init__(self, id2label=None):
        self.config = types.SimpleNamespace(id2label=id2label or {0: "SUBSTANTIVE", 1: "WEAK", 2: "NO_CLAIM"})
        self.trained_steps = 0

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls(kw.get("id2label"))

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def parameters(self):
        return []

    def __call__(self, input_ids, labels=None):
        if labels is not None:
            self.trained_steps += 1
            return types.SimpleNamespace(loss=types.SimpleNamespace(backward=lambda: None))
        # long inputs -> SUBSTANTIVE, short -> NO_CLAIM
        rows = [[2.0, 0.0, 0.0] if n[0] > 40 else [0.0, 0.0, 2.0] for n in input_ids.data]
        return types.SimpleNamespace(logits=T(rows))

    def save_pretrained(self, out):
        Path(out, "config.json").write_text(json.dumps({"id2label": self.config.id2label}))


def fake_modules():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.no_grad = NoGrad
    torch.manual_seed = lambda s: None
    torch.tensor = lambda x: T(x)
    optim = types.SimpleNamespace(step=lambda: None, zero_grad=lambda: None)
    torch.optim = types.SimpleNamespace(AdamW=lambda params, lr: optim)
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = FakeTokenizer
    tr.AutoModelForSequenceClassification = FakeModel
    tr.get_linear_schedule_with_warmup = lambda opt, warm, steps: types.SimpleNamespace(step=lambda: None)
    return {"torch": torch, "transformers": tr}


def cand(text, prev="", nxt=""):
    return Candidate(
        filing_id="f",
        sentence_id=text[:10],
        section="10-K:1",
        text=text,
        prev_text=prev,
        next_text=nxt,
        firm_name="Fictional Corp.",
        terms=["ai_acronym"],
    )


class TransformerBackendTests(unittest.TestCase):
    def test_inference(self):
        with mock.patch.dict(sys.modules, fake_modules()):
            from aidisclose.classify import get_classifier

            clf = get_classifier("transformer", model_path="/models/x", batch_size=2)
            preds = clf.classify(
                [cand("We use AI to detect fraud in card payments."), cand("AI."), cand("AI is used.")]
            )
            self.assertEqual(
                [p.state for p in preds], [AdoptionState.SUBSTANTIVE, AdoptionState.NO_CLAIM, AdoptionState.NO_CLAIM]
            )
            self.assertEqual(preds[0].reason, Reason.UNSPECIFIED)
            self.assertAlmostEqual(sum(preds[0].details["probabilities"].values()), 1.0, places=3)
            first, second, kw = FakeTokenizer.calls[-2]
            self.assertTrue(first[0].startswith("Fictional Corp. | 10-K:1 | "))
            self.assertEqual(second[0], "(no context)")
            self.assertTrue(kw["truncation"])
            self.assertTrue(clf.config()["experimental"])
            with self.assertRaises(ValueError):
                get_classifier("transformer")

    def test_wrong_labels_rejected(self):
        with mock.patch.dict(sys.modules, fake_modules()):
            from aidisclose.classify.transformer import TransformerClassifier

            with mock.patch.object(
                FakeModel, "from_pretrained", classmethod(lambda cls, p, **k: cls({0: "POS", 1: "NEG"}))
            ):
                with self.assertRaises(ValueError):
                    TransformerClassifier("/models/x")

    def test_train_writes_metadata(self):
        with mock.patch.dict(sys.modules, fake_modules()), tempfile.TemporaryDirectory() as tmp:
            from aidisclose.classify.transformer import TransformerClassifier

            cands = [cand(f"We use AI number {i}.") for i in range(5)]
            labels = ["SUBSTANTIVE", "WEAK", "NO_CLAIM", "WEAK", "substantive"]
            out = TransformerClassifier.train(cands, labels, Path(tmp) / "m", epochs=2, batch_size=2, base_model="base")
            meta = json.loads((out / "aidisclose_training.json").read_text())
            self.assertEqual(meta["n_train"], 5)
            self.assertEqual(meta["label_counts"], {"SUBSTANTIVE": 2, "WEAK": 2, "NO_CLAIM": 1})
            self.assertEqual(len(meta["train_data_sha256"]), 64)
            with self.assertRaises(ValueError):
                TransformerClassifier.train(cands, ["WEAK"] * 5, Path(tmp) / "m2")
            with self.assertRaises(ValueError):
                TransformerClassifier.train(cands, ["MAYBE"] * 4 + ["WEAK"], Path(tmp) / "m3")
            clf = TransformerClassifier(out)
            self.assertIn("training", clf.config())


if __name__ == "__main__":
    unittest.main()
