"""Classifier backends.

=============  ==============================================================
``rules``      Deterministic cue-based rules (default, no extra dependencies)
``supervised`` TF-IDF + logistic regression trained on your annotations
               (``pip install aidisclose[ml]``)
``llm``        Any OpenAI-compatible chat-completions endpoint; intended for
               locally served open-weight models (vLLM, Ollama, llama.cpp)
``transformer`` EXPERIMENTAL fine-tuned sequence classifier
               (``pip install aidisclose[transformers]``)
=============  ==============================================================
"""

from __future__ import annotations

from typing import Any

from .base import Classifier
from .rules import RuleClassifier


def get_classifier(name: str, **options: Any) -> Classifier:
    key = name.lower()
    if key in {"rules", "rule", "rule-based"}:
        return RuleClassifier(**options)
    if key in {"supervised", "sklearn"}:
        from .supervised import SupervisedClassifier

        model_path = options.pop("model_path", None)
        if model_path is None:
            raise ValueError("the supervised backend needs model_path=<trained model file>")
        return SupervisedClassifier.load(model_path)
    if key in {"llm", "openai", "openai-compatible"}:
        from .llm import LLMClassifier

        return LLMClassifier(**options)
    if key in {"transformer", "transformers"}:
        from .transformer import TransformerClassifier

        model_path = options.pop("model_path", None)
        if model_path is None:
            raise ValueError("the transformer backend needs model_path=<fine-tuned model directory>")
        return TransformerClassifier(model_path, **options)
    raise ValueError(f"unknown backend {name!r}; choose rules, supervised, llm or transformer")


__all__ = ["Classifier", "RuleClassifier", "get_classifier"]
