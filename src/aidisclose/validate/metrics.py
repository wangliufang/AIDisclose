"""Agreement and accuracy metrics (dependency-free, sample-weight aware).

All functions accept optional ``weights``: inverse inclusion probabilities from
stratified sampling (``sampling_weight`` in the key file written by
``aidisclose sample``). With weights, every estimate -- precision, recall,
F1, accuracy, Cohen's kappa and the bootstrap confidence intervals -- refers
to the population of candidate sentences rather than to the over-sampled
annotation sample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..models import STATES

BINARY_VIEWS = (("substantive_vs_rest", ("SUBSTANTIVE",)), ("claim_vs_no_claim", ("SUBSTANTIVE", "WEAK")))


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = STATES, weights: Sequence[float] | None = None
) -> np.ndarray:
    """Rows are true labels, columns are predicted labels (cells hold summed weights)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    if len(idx) != len(labels) or len(labels) == 0:
        raise ValueError("labels must be nonempty and unique")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if any(t not in idx for t in y_true) or any(p not in idx for p in y_pred):
        raise ValueError("all labels must belong to the declared label set")
    m = np.zeros((len(labels), len(labels)), dtype=float)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(y_true) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite, positive and match the label length")
    for t, p, wi in zip(y_true, y_pred, w):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += wi
    return m


def _prf(m: np.ndarray, labels: Sequence[str]) -> dict[str, dict[str, float]]:
    out = {}
    for i, lab in enumerate(labels):
        tp = m[i, i]
        fp = m[:, i].sum() - tp
        fn = m[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        # A class that occurs in either truth or predictions has defined F1;
        # an entirely absent class is excluded from the macro average.
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else float("nan")
        out[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": float(m[i, :].sum())}
    return out


def _kappa_from_matrix(m: np.ndarray) -> float:
    n = m.sum()
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / (n * n)
    if pe == 1:
        return float("nan")
    return float((po - pe) / (1 - pe))


def cohen_kappa(
    a: Sequence[str], b: Sequence[str], labels: Sequence[str] | None = None, weights: Sequence[float] | None = None
) -> float:
    """Cohen's (1960) kappa for two raters; ``weights`` are sampling weights, not disagreement weights."""
    if len(a) == 0 and len(b) == 0 and labels is None:
        if weights is not None and len(weights) != 0:
            raise ValueError("weights must match the label length")
        return float("nan")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b))
    return _kappa_from_matrix(confusion_matrix(a, b, labels, weights))


def collapse(labels: Sequence[str], positive: Sequence[str]) -> list[str]:
    pos = set(positive)
    return ["POS" if lab in pos else "NEG" for lab in labels]


def _report_from_matrix(m: np.ndarray, labels: Sequence[str], binary: dict[str, np.ndarray]) -> dict[str, Any]:
    per = _prf(m, labels)
    f1s = [v["f1"] for v in per.values() if not np.isnan(v["f1"])]
    rep: dict[str, Any] = {
        "accuracy": float(np.trace(m) / m.sum()) if m.sum() else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "kappa": _kappa_from_matrix(m),
        "per_class": per,
    }
    for name, bm in binary.items():
        rep[name] = _prf(bm, ["POS", "NEG"])["POS"]
    return rep


def _matrices(y_true, y_pred, labels, weights):
    m = confusion_matrix(y_true, y_pred, labels, weights)
    binary = {
        name: confusion_matrix(collapse(y_true, pos), collapse(y_pred, pos), ["POS", "NEG"], weights)
        for name, pos in BINARY_VIEWS
    }
    return m, binary


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = STATES,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Per-class precision/recall/F1, macro-F1, accuracy, kappa and two binary views.

    The binary views answer the questions that matter for panel indicators:
    *substantive vs. rest* (``substantive_any``) and *claim vs. no claim*
    (``claim_any``).
    """
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]
    # Validate against the multiclass labels before constructing binary views.
    confusion_matrix(y_true, y_pred, labels, weights)
    m, binary = _matrices(y_true, y_pred, labels, weights)
    rep = {"n": len(y_true), "weighted": weights is not None}
    rep.update(_report_from_matrix(m, labels, binary))
    rep["confusion_matrix"] = {"labels": list(labels), "matrix": m.round(3).tolist()}
    return rep


def _flatten(rep: dict[str, Any]) -> dict[str, float]:
    flat = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"], "kappa": rep["kappa"]}
    for lab, v in rep["per_class"].items():
        for k in ("precision", "recall", "f1"):
            flat[f"{lab}.{k}"] = v[k]
    for name, _ in BINARY_VIEWS:
        for k in ("precision", "recall", "f1"):
            flat[f"{name}.{k}"] = rep[name][k]
    return flat


def bootstrap_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = STATES,
    weights: Sequence[float] | None = None,
    strata: Sequence[str] | None = None,
    clusters: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Percentile bootstrap CIs for every scalar in :func:`classification_report`.

    Rows are resampled with replacement (within ``strata`` if given). When
    ``clusters`` is supplied, whole clusters are sampled together and this
    takes precedence over row strata. Sampling weights travel with each row.
    Returns ``{metric: (point, lower, upper)}``.
    """
    yt = [str(y) for y in y_true]
    yp = [str(y) for y in y_pred]
    n = len(yt)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != n:
        raise ValueError("weights must have the same length as y_true")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if strata is not None and len(strata) != n:
        raise ValueError("strata must have the same length as y_true")
    if clusters is not None and len(clusters) != n:
        raise ValueError("clusters must have the same length as y_true")
    rep_clusters = None if clusters is None else [str(c) for c in clusters]
    rep_strata = None if strata is None else [str(s) for s in strata]
    if rep_clusters is not None:
        if any(value.strip().lower() in {"", "nan", "none", "<na>"} for value in rep_clusters):
            raise ValueError("clusters cannot contain missing identifiers")
        if len(set(rep_clusters)) < 2:
            raise ValueError("cluster bootstrap needs at least two distinct clusters")
    if (
        rep_strata is not None
        and rep_clusters is None
        and any(value.strip().lower() in {"", "nan", "none", "<na>"} for value in rep_strata)
    ):
        raise ValueError("strata cannot contain missing identifiers")
    point = _flatten(classification_report(yt, yp, labels, None if weights is None else w))
    if n == 0:
        return {k: (v, float("nan"), float("nan")) for k, v in point.items()}
    rng = np.random.default_rng(seed)
    stratum_labels = ["__all__"] * n if rep_strata is None or rep_clusters is not None else rep_strata
    strata_groups = {
        label: np.flatnonzero(np.asarray(stratum_labels, dtype=object) == label)
        for label in dict.fromkeys(stratum_labels)
    }
    # A singleton stratum cannot supply a nonparametric estimate of its
    # within-stratum variability without knowing it is a census.
    if rep_clusters is None and any(len(group) < 2 for group in strata_groups.values()):
        return {k: (float(v), float("nan"), float("nan")) for k, v in point.items()}
    if rep_clusters is None:
        groups = list(strata_groups.values())
        cluster_groups = None
    else:
        cluster_groups = {
            label: [
                ix[np.flatnonzero(np.asarray(rep_clusters, dtype=object)[ix] == c)]
                for c in dict.fromkeys(rep_clusters[i] for i in ix)
            ]
            for label, ix in strata_groups.items()
        }
    yt_a = np.asarray(yt, dtype=object)
    yp_a = np.asarray(yp, dtype=object)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        if cluster_groups is None:
            ix = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        else:
            chosen = []
            for clusters_in_stratum in cluster_groups.values():
                selected = rng.integers(0, len(clusters_in_stratum), len(clusters_in_stratum))
                chosen.extend(clusters_in_stratum[j] for j in selected)
            ix = np.concatenate(chosen)
        m, binary = _matrices(list(yt_a[ix]), list(yp_a[ix]), labels, w[ix])
        flat = _flatten(_report_from_matrix(m, labels, binary))
        for k, v in flat.items():
            if not np.isnan(v):
                samples[k].append(v)
    out = {}
    for k, pt in point.items():
        vals = samples[k]
        if vals:
            lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
            out[k] = (float(pt), float(lo), float(hi))
        else:
            out[k] = (float(pt), float("nan"), float("nan"))
    return out


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI for an arbitrary unweighted *metric*."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    n = len(yt)
    stats = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        v = metric(list(yt[ix]), list(yp[ix]))
        if not np.isnan(v):
            stats.append(v)
    point = metric(list(yt), list(yp))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def report_to_markdown(rep: dict[str, Any], title: str = "Validation report") -> str:
    def f(x: float | None) -> str:
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

    ci = rep.get("ci", {})

    def fc(key: str, value: float) -> str:
        if key in ci:
            _, lo, hi = ci[key]
            return f"{f(value)} [{f(lo)}, {f(hi)}]"
        return f(value)

    lines = [
        f"## {title}",
        "",
        f"n = {rep['n']}" + (" (sampling-weighted estimates)" if rep.get("weighted") else ""),
        "",
    ]
    if ci:
        method = "cluster bootstrap" if rep.get("ci_method") == "cluster" else "percentile bootstrap"
        lines += [f"Values in brackets are 95% {method} confidence intervals.", ""]
    lines += ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab, v in rep["per_class"].items():
        lines.append(
            f"| {lab} | {fc(lab + '.precision', v['precision'])} | {fc(lab + '.recall', v['recall'])} | "
            f"{fc(lab + '.f1', v['f1'])} | {v['support']:.0f} |"
        )
    lines += [
        "",
        f"Accuracy: {fc('accuracy', rep['accuracy'])}  |  Macro-F1: {fc('macro_f1', rep['macro_f1'])}  |  "
        f"Cohen's kappa: {fc('kappa', rep['kappa'])}",
        "",
    ]
    for name, label in (("substantive_vs_rest", "SUBSTANTIVE vs rest"), ("claim_vs_no_claim", "Claim vs no claim")):
        v = rep[name]
        lines.append(
            f"{label}: precision {fc(name + '.precision', v['precision'])}, recall {fc(name + '.recall', v['recall'])}, "
            f"F1 {fc(name + '.f1', v['f1'])}"
        )
    return "\n".join(lines) + "\n"
