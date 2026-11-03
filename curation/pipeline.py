"""Training and prediction of one task: image model + metadata model -> fusion -> rules."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from . import image_model, meta_model
from .config import paths
from .fusion import decide, fuse
from .labels import patient_split, read_labels, task_rows
from .rules import RuleSet
from .volumes import VolumeLoader

log = logging.getLogger(__name__)


def volume_loader(cfg: dict, index) -> VolumeLoader:
    return VolumeLoader(index, cfg["volume"]["n_slices"], cfg["volume"]["size"], paths(cfg)["cache"])


def model_dir(cfg: dict, task: str) -> str:
    return os.path.join(paths(cfg)["models"], task)


def training_rows(cfg: dict, task_name: str, index, exclude_patients=()) -> dict:
    """{"image": rows, "meta": rows}: each model's training rows from its label sources
    (task `train_labels`), without the patients in exclude_patients."""
    task, out = cfg["tasks"][task_name], {}
    for m in ("image", "meta"):
        rows = task_rows(cfg, task_name, index, read_labels(cfg, task["train_labels"][m]))
        out[m] = rows[~rows["patient"].isin(set(exclude_patients))].reset_index(drop=True)
    return out


def train_models(cfg: dict, task_name: str, index, rows: dict, out_dir: str) -> dict:
    """Both models of a task (rows: training_rows). The image model keeps a patient-level
    validation split for early stopping and temperature; the metadata model uses all its rows."""
    task = cfg["tasks"][task_name]
    classes = list(task["classes"])
    os.makedirs(out_dir, exist_ok=True)
    img = rows["image"]
    tr, va = patient_split(img["patient"].tolist(), cfg["image_model"]["val_fraction"], cfg["seed"])
    log.info("task %s: image model on %d rows, validation %d rows; metadata model on %d rows",
             task_name, len(tr), len(va), len(rows["meta"]))
    info = image_model.train(img.iloc[tr].reset_index(drop=True), img.iloc[va].reset_index(drop=True),
                             classes, task, cfg, volume_loader(cfg, index), os.path.join(out_dir, "image_model.pt"))
    meta_model.train(index, rows["meta"]["key"].tolist(), rows["meta"]["label"].tolist(), classes,
                     os.path.join(out_dir, "meta_model.joblib"), cfg["meta_model"]["text_dim"], cfg["seed"])
    info.update(task=task_name, classes=classes, n_rows_image=len(img), n_rows_meta=len(rows["meta"]),
                n_patients=int(img["patient"].nunique()), label_sources=task["train_labels"])
    with open(os.path.join(out_dir, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def predict_probs(cfg: dict, index, keys: list[str], out_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(image probabilities, metadata probabilities) for keys, both in the task's class order."""
    loader = volume_loader(cfg, index)
    p_img, _ = image_model.predict(os.path.join(out_dir, "image_model.pt"), keys, loader,
                                   cfg["image_model"]["batch_size"], cfg["image_model"]["workers"])
    p_meta = meta_model.predict(os.path.join(out_dir, "meta_model.joblib"), index, keys)
    return p_img, p_meta


def label_keys(cfg: dict, task_name: str, index, keys: list[str], out_dir: str, weight: float | None = None) -> pd.DataFrame:
    """Final labels of keys: key, label, confidence, source, reason, uncertain."""
    task = cfg["tasks"][task_name]
    if not keys:
        return pd.DataFrame(columns=["key", "label", "confidence", "source", "reason", "uncertain"])
    p_img, p_meta = predict_probs(cfg, index, keys, out_dir)
    w = task["fusion_weight"] if weight is None else weight
    d = decide(keys, fuse(p_img, p_meta, w), list(task["classes"]), RuleSet(task["rules"], task["classes"]), index)
    d["uncertain"] = d["confidence"] < task["threshold"]
    return d
