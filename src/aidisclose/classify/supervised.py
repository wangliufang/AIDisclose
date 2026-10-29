"""Supervised backend trained on human-annotated sentences.

The model is a word and character n-gram TF-IDF representation of the
sentence, augmented with tokens that encode the section and the cues found by
the rule backend (``__agent_focal__``, ``__forward__``, ...), fed to a
class-balanced multinomial logistic regression. It is small enough to train on
a laptop from a few hundred annotated sentences and gives a reproducible,
fully open alternative to prompting a language model.

Requires ``pip install aidisclose[ml]`` (scikit-learn, joblib).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import AdoptionState, Candidate, Prediction, Reason
from .base import Classifier
from .rules import RuleClassifier

_CUE_KEYS = (
    "agent",
    "negation",
    "regulatory",
    "market",
    "deployment",
    "offer",
    "investment",
    "forward",
    "continue",
    "hedge",
    "risk",
    "product_adj",
    "use_noun",
    "possessive",
    "application",
    "quantity",
    "date",
    "named_product",
    "specific",
    "service_to_third_party",
    # rules 1.1.0
    "licensed_in",
    "agentless_use",
    "effect_prediction",
    "financial",
    "third_party_supplied",
)


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("the supervised backend requires 'pip install aidisclose[ml]'") from exc


def featurize(cand: Candidate, rules: RuleClassifier | None = None) -> str:
    """Text representation used by the supervised model."""
    rules = rules or RuleClassifier()
    pred = rules.classify_one(cand)
    tokens = [f"__section_{cand.section.replace(':', '_').replace('-', '')}__"]
    agent = pred.details.get("agent", [])
    if agent:
        tokens.append("__agent_present__")
    for key in _CUE_KEYS:
        if key in pred.details:
            tokens.append(f"__{key}__")
    tokens.append(f"__rule_{pred.state.value.lower()}__")
    return " ".join(tokens) + " " + cand.text


class SupervisedClassifier(Classifier):
    name = "supervised"

    def __init__(self, pipeline=None, meta: dict[str, Any] | None = None):
        self.pipeline = pipeline
        self.meta = meta or {}
        self._rules = RuleClassifier()

    # ---------------------------------------------------------------- training
    @staticmethod
    def _make_pipeline(C: float = 4.0, random_state: int = 0):
        _require_sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
            ]
        )
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=random_state)
        return Pipeline([("features", features), ("clf", clf)])

    @classmethod
    def train(
        cls,
        candidates: Sequence[Candidate],
        labels: Sequence[str],
        sample_weight: Sequence[float] | None = None,
        C: float = 4.0,
        random_state: int = 0,
    ) -> SupervisedClassifier:
        if len(candidates) != len(labels):
            raise ValueError("candidates and labels must have the same length")
        y = [AdoptionState(str(lbl).upper()).value for lbl in labels]
        if len(set(y)) < 2:
            raise ValueError("training data must contain at least two classes")
        rules = RuleClassifier()
        X = [featurize(c, rules) for c in candidates]
        pipe = cls._make_pipeline(C=C, random_state=random_state)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = list(sample_weight)
        pipe.fit(X, y, **fit_kwargs)
        import sklearn

        meta = {
            "n_train": len(y),
            "class_counts": {s: y.count(s) for s in sorted(set(y))},
            "C": C,
            "random_state": random_state,
            "sklearn_version": sklearn.__version__,
            "weighted_fit": sample_weight is not None,
        }
        return cls(pipe, meta)

    @classmethod
    def cross_validate(
        cls,
        candidates: Sequence[Candidate],
        labels: Sequence[str],
        k: int = 5,
        random_state: int = 0,
        C: float = 4.0,
        weights: Sequence[float] | None = None,
        strata: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        n_boot: int = 1000,
    ) -> dict[str, Any]:
        """Stratified k-fold estimate of out-of-sample performance.

        Every sentence is predicted by a model that did not see it. The report
        has the same fields as :func:`~aidisclose.validate.classification_report`
        (sampling-weighted if ``weights`` are given), bootstrap confidence
        intervals under ``"ci"`` and the out-of-fold predictions under
        ``"oof_predictions"``.
        """
        _require_sklearn()
        from sklearn.model_selection import StratifiedKFold

        from ..validate.metrics import bootstrap_report, classification_report

        y = [AdoptionState(str(lbl).upper()).value for lbl in labels]
        if len(candidates) != len(y):
            raise ValueError("candidates and labels must have the same length")
        if weights is not None and len(weights) != len(y):
            raise ValueError("weights must have the same length as labels")
        if groups is not None and len(groups) != len(y):
            raise ValueError("groups must have the same length as labels")
        idx = list(range(len(y)))
        if groups is None:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            folds = splitter.split(idx, y)
        else:
            from sklearn.model_selection import StratifiedGroupKFold

            if len(set(str(g) for g in groups)) < k:
                raise ValueError(f"grouped {k}-fold CV requires at least {k} distinct groups")
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
            folds = splitter.split(idx, y, groups)
        preds: list[str | None] = [None] * len(y)
        for train_idx, test_idx in folds:
            model = cls.train(
                [candidates[i] for i in train_idx],
                [y[i] for i in train_idx],
                C=C,
                random_state=random_state,
                sample_weight=None if weights is None else [weights[i] for i in train_idx],
            )
            out = model.classify([candidates[i] for i in test_idx])
            for i, p in zip(test_idx, out):
                preds[i] = p.state.value
        report = classification_report(y, preds, weights=weights)
        report["ci"] = bootstrap_report(y, preds, weights=weights, strata=strata, clusters=groups, n_boot=n_boot)
        report["k"] = k
        report["cv_grouped"] = groups is not None
        report["ci_method"] = "cluster" if groups is not None else "stratified_rows" if strata is not None else "row"
        report["oof_predictions"] = preds
        return report

    # --------------------------------------------------------------- inference
    def classify(self, candidates: Sequence[Candidate]) -> list[Prediction]:
        if self.pipeline is None:
            raise RuntimeError("model is not trained; use SupervisedClassifier.train or .load")
        if not candidates:
            return []
        X = [featurize(c, self._rules) for c in candidates]
        probs = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        out = []
        for c, row in zip(candidates, probs):
            j = int(row.argmax())
            state = AdoptionState(classes[j])
            rule_pred = self._rules.classify_one(c)
            reason = rule_pred.reason if rule_pred.state == state else Reason.UNSPECIFIED
            out.append(
                Prediction(
                    state,
                    reason,
                    float(row[j]),
                    self.name,
                    {"probabilities": {cl: round(float(p), 4) for cl, p in zip(classes, row)}},
                )
            )
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path)
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SupervisedClassifier:
        """Load a model saved with :meth:`save`.

        Model files are pickles: only load files you created or trust.
        """
        _require_sklearn()
        import joblib

        obj = joblib.load(Path(path))
        return cls(obj["pipeline"], obj.get("meta", {}))

    def config(self) -> dict[str, Any]:
        # Hash prediction-relevant fitted state in a serialization-independent
        # representation; joblib's object hash may change on save/load.
        digest = None
        if self.pipeline is not None:
            try:
                features = self.pipeline.named_steps["features"]
                classifier = self.pipeline.named_steps["clf"]
                vectors = []
                for name, vector in features.transformer_list:
                    vectors.append(
                        {
                            "name": name,
                            "params": vector.get_params(),
                            "vocabulary": vector.vocabulary_,
                            "idf": vector.idf_.tolist(),
                        }
                    )
                state = {
                    "vectors": vectors,
                    "transformer_weights": features.transformer_weights,
                    "classifier_params": classifier.get_params(),
                    "classes": classifier.classes_.tolist(),
                    "coef": classifier.coef_.tolist(),
                    "intercept": classifier.intercept_.tolist(),
                    "rule_config": self._rules.config(),
                }
                payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                digest = hashlib.sha256(payload).hexdigest()
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError("unsupported supervised model: cannot fingerprint fitted state") from exc
        return {"backend": self.name, **self.meta, "fitted_model_sha256": digest}
