"""EXPERIMENTAL fine-tuned transformer backend (optional; ``pip install aidisclose[transformers]``).

This backend classifies candidate sentences with a sequence-classification
model fine-tuned on *your* annotations (for example a DeBERTa or RoBERTa
checkpoint). It predicts the adoption state only (the reason is
``UNSPECIFIED``).

Status: the inference and training code paths are exercised in the test suite
with stand-in objects only; the AIDisclose authors have not yet benchmarked a
fine-tuned model. Validate any model you train with ``aidisclose evaluate`` on
a held-out annotated sample before using it for measurement.

The model input is ``"<firm> | <section> | <target sentence>"`` paired with
``"<previous sentence> <next sentence>"``, so that context can resolve the
agent of passive sentences, as in the codebook.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import STATES, AdoptionState, Candidate, Prediction, Reason
from .base import Classifier, batched

_INSTALL_HINT = "the transformer backend needs 'transformers' and 'torch': pip install aidisclose[transformers]"


def _import():
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_INSTALL_HINT) from exc
    return torch, transformers


def model_inputs(c: Candidate) -> tuple[str, str]:
    first = f"{c.firm_name or 'unknown firm'} | {c.section} | {c.text}"
    second = " ".join(x for x in (c.prev_text, c.next_text) if x) or "(no context)"
    return first, second


class TransformerClassifier(Classifier):
    """Classify with a fine-tuned sequence-classification model (experimental).

    Parameters
    ----------
    model_path:
        Directory written by :meth:`train` or any Hugging Face model whose
        ``id2label`` are the three states.
    device:
        ``"cpu"``, ``"cuda"`` ... (default: CUDA if available).
    """

    name = "transformer"

    def __init__(self, model_path: str | Path, device: str | None = None, batch_size: int = 16, max_length: int = 256):
        torch, transformers = _import()
        self._torch = torch
        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        id2label = {int(k): str(v).upper() for k, v in dict(self.model.config.id2label).items()}
        if sorted(id2label.values()) != sorted(STATES):
            raise ValueError(f"model labels {sorted(id2label.values())} are not the AIDisclose states {STATES}")
        self.id2label = id2label

    def config(self) -> dict[str, Any]:
        cfg = {"backend": self.name, "model_path": self.model_path, "max_length": self.max_length, "experimental": True}
        meta = Path(self.model_path) / "aidisclose_training.json"
        if meta.exists():
            cfg["training"] = json.loads(meta.read_text(encoding="utf-8"))
        return cfg

    def classify(self, candidates: Sequence[Candidate]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(list(candidates), self.batch_size):
            firsts, seconds = zip(*(model_inputs(c) for c in batch))
            with self._torch.no_grad():
                enc = self.tokenizer(
                    list(firsts),
                    list(seconds),
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                probs = self.model(**enc).logits.softmax(-1).tolist()
            for p in probs:
                k = max(range(len(p)), key=p.__getitem__)
                state = AdoptionState(self.id2label[k])
                details = {"probabilities": {self.id2label[i]: round(float(x), 4) for i, x in enumerate(p)}}
                out.append(Prediction(state, Reason.UNSPECIFIED, float(p[k]), self.name, details))
        return out

    # ------------------------------------------------------------- training
    @staticmethod
    def train(
        candidates: Sequence[Candidate],
        labels: Sequence[str],
        out_dir: str | Path,
        base_model: str = "microsoft/deberta-v3-base",
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 256,
        seed: int = 42,
        device: str | None = None,
    ) -> Path:
        """Fine-tune *base_model* on annotated candidates and save it to *out_dir*.

        A plain PyTorch loop (AdamW, linear decay, no early stopping); the
        training settings and a digest of the training data are written to
        ``aidisclose_training.json`` next to the model.
        """
        torch, transformers = _import()
        labels = [str(x).upper() for x in labels]
        if len(candidates) != len(labels) or not labels:
            raise ValueError("need one label per candidate")
        if len(set(labels)) < 2:
            raise ValueError("need at least two different labels")
        bad = sorted(set(labels) - set(STATES))
        if bad:
            raise ValueError(f"unknown labels {bad}")
        random.seed(seed)
        torch.manual_seed(seed)
        label2id = {s: i for i, s in enumerate(STATES)}
        tok = transformers.AutoTokenizer.from_pretrained(base_model)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(STATES), id2label=dict(enumerate(STATES)), label2id=label2id
        )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        pairs = [model_inputs(c) for c in candidates]
        y = [label2id[x] for x in labels]
        order = list(range(len(y)))
        steps = max(1, epochs * ((len(y) + batch_size - 1) // batch_size))
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        sched = transformers.get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
        model.train()
        for _ in range(epochs):
            random.shuffle(order)
            for idx in batched(order, batch_size):
                enc = tok(
                    [pairs[i][0] for i in idx],
                    [pairs[i][1] for i in idx],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                target = torch.tensor([y[i] for i in idx]).to(device)
                loss = model(**enc, labels=target).loss
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        digest = hashlib.sha256(json.dumps([[p[0], p[1], lab] for p, lab in zip(pairs, labels)]).encode()).hexdigest()
        meta = {
            "base_model": base_model,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length,
            "seed": seed,
            "n_train": len(labels),
            "label_counts": {s: labels.count(s) for s in STATES},
            "train_data_sha256": digest,
        }
        (out / "aidisclose_training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out
