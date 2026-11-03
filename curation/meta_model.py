"""Metadata model: gradient-boosted trees (HistGradientBoosting, balanced sample weights) on the 46
header features plus the LSA text encoding of the description. The text encoder is fitted on the
training rows only and stored in the same file as the trees."""
from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .features import TextEncoder, header_matrix, texts


def _matrix(index, keys, encoder: TextEncoder) -> np.ndarray:
    return np.hstack([header_matrix(index, keys), encoder.transform(texts(index, keys))])


def train(index, keys: list[str], labels: list[str], classes: list[str], out_path: str,
          text_dim: int = 48, seed: int = 42) -> None:
    enc = TextEncoder(dim=text_dim, seed=seed).fit(texts(index, keys))
    y = np.array([classes.index(l) for l in labels])
    est = HistGradientBoostingClassifier(random_state=seed)
    est.fit(_matrix(index, keys, enc), y, sample_weight=compute_sample_weight("balanced", y))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"estimator": est, "encoder": enc, "classes": classes}, out_path)


def predict(path: str, index, keys: list[str]) -> np.ndarray:
    """Probabilities (len(keys), n_classes) in the stored class order (a class unseen in training: 0)."""
    b = joblib.load(path)
    proba = b["estimator"].predict_proba(_matrix(index, keys, b["encoder"]))
    full = np.zeros((len(keys), len(b["classes"])))
    for col, ci in enumerate(b["estimator"].classes_):
        full[:, int(ci)] = proba[:, col]
    return full
