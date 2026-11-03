"""Late fusion of the two models and the final label of each image group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import Header, RuleSet


def fuse(p_image: np.ndarray, p_meta: np.ndarray, w: float) -> np.ndarray:
    """Weighted geometric mean p_image^w * p_meta^(1-w), renormalised per row. Rows without an image
    prediction (NaN: volume not loadable) fall back to the metadata model."""
    p_image = np.asarray(p_image, float)
    p_meta = np.asarray(p_meta, float)
    s = np.exp(w * np.log(np.clip(p_image, 1e-6, None)) + (1 - w) * np.log(np.clip(p_meta, 1e-6, None)))
    s = s / s.sum(axis=1, keepdims=True)
    missing = np.isnan(p_image).any(axis=1)
    s[missing] = p_meta[missing]
    return s


def decide(keys: list[str], probs: np.ndarray, classes: list[str], rules: RuleSet, index) -> pd.DataFrame:
    """key, label, confidence, source (rule / model / model+mask), reason for each image group."""
    rows = []
    for k, p in zip(keys, probs):
        label, conf, source, why = rules.apply(p, classes, Header.from_entry(index.get(k), index))
        rows.append({"key": k, "label": label, "confidence": conf, "source": source, "reason": why})
    return pd.DataFrame(rows)


def choose_weight(y: list[str], p_image: np.ndarray, p_meta: np.ndarray, classes: list[str],
                  rules: RuleSet, keys: list[str], index, metric: str = "macro_f1",
                  grid=np.round(np.arange(0.1, 0.95, 0.1), 1)) -> tuple[float, dict]:
    """Fusion weight with the best metric (after the rules) on out-of-fold predictions."""
    from .stats import METRICS
    scores = {}
    for w in grid:
        d = decide(keys, fuse(p_image, p_meta, w), classes, rules, index)
        scores[float(w)] = METRICS[metric](np.asarray(y), d["label"].to_numpy())
    return max(scores, key=scores.get), scores
