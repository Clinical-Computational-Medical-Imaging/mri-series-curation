"""Patient-level bootstrap statistics and the confidence gate.

METRICS                                       macro_f1, balanced_accuracy (mean recall over the classes present)
bootstrap(y, pred, patients, metric)          metric with a 95 % CI (patients resampled with all their rows)
paired_bootstrap(y, pred_a, pred_b, patients, metric)
                                              difference a - b on the same resamples, CI and two-sided p
threshold_for_budget(...)                     lowest confidence threshold whose error among accepted rows
                                              stays within a budget; safe=True uses the bootstrap 95 %
                                              upper bound of that error instead of the point estimate
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score

B, SEED, ALPHA = 2000, 42, 0.05


def _resamples(patients, b: int = B, seed: int = SEED):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(patients), return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    for _ in range(b):
        yield np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])


def macro_f1(y, p) -> float:
    return f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)


def balanced_accuracy(y, p) -> float:
    return recall_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)


METRICS = {"macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy}


def bootstrap(y, pred, patients, metric: str = "macro_f1", b: int = B, seed: int = SEED) -> tuple[float, float, float]:
    f, y, pred = METRICS[metric], np.asarray(y), np.asarray(pred)
    vals = [f(y[i], pred[i]) for i in _resamples(patients, b, seed)]
    lo, hi = np.quantile(vals, [ALPHA / 2, 1 - ALPHA / 2])
    return f(y, pred), float(lo), float(hi)


def paired_bootstrap(y, pred_a, pred_b, patients, metric: str = "macro_f1", b: int = B,
                     seed: int = SEED) -> tuple[float, float, float, float]:
    f, y, pa, pb = METRICS[metric], np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    d = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in _resamples(patients, b, seed)])
    lo, hi = np.quantile(d, [ALPHA / 2, 1 - ALPHA / 2])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return f(y, pa) - f(y, pb), float(lo), float(hi), float(p)


def coverage(y, pred, conf, threshold: float) -> tuple[float, float]:
    """(share accepted, error among accepted) at a confidence threshold."""
    acc = np.asarray(conf) >= threshold
    wrong = np.asarray(y) != np.asarray(pred)
    return float(acc.mean()), float(wrong[acc].mean()) if acc.any() else 0.0


def threshold_for_budget(y, pred, conf, patients, budget: float, safe: bool = True,
                         n_boot: int = 500, seed: int = SEED) -> float:
    wrong = (np.asarray(y) != np.asarray(pred)).astype(float)
    conf = np.asarray(conf, float)
    boots = list(_resamples(patients, n_boot, seed)) if safe else [np.arange(len(conf))]
    for t in np.round(np.arange(0.30, 1.0001, 0.005), 3):
        errs = []
        for i in boots:
            sel = i[conf[i] >= t]
            errs.append(wrong[sel].mean() if len(sel) else 0.0)
        if (np.quantile(errs, 0.95) if safe else errs[0]) <= budget:
            return float(t)
    return 1.0
