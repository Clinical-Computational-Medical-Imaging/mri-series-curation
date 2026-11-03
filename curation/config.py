"""Configuration: one YAML file (see config.example.yaml). Everything dataset- or task-specific
lives there: input, label files, tasks and their classes, how image groups inherit labels, which
header rules apply, localizer thresholds, training settings."""
from __future__ import annotations

import copy
import os

import yaml

DEFAULTS = {
    "input": {"kind": "folder", "root": None},
    "work_dir": "work",
    "labels": {},           # name -> label CSV, e.g. {expert: labels/expert.csv, reviewed: labels/reviewed.csv}
    "patient_column": "patient_id",
    "seed": 42,
    "workers": 8,
    "localizer": {"max_slices": 5, "max_matrix": 256, "max_slices_without_matrix": 3,
                  "keywords": ["localizer", "localiser", "scout", "survey", "3-plane loc"]},
    "volume": {"n_slices": 16, "size": 224},
    "image_model": {"backbone": "convnext_tiny", "epochs": 60, "batch_size": 16, "lr": 3.0e-4,
                    "patience": 15, "val_fraction": 0.2, "workers": 8},
    "meta_model": {"text_dim": 48},
    "tasks": {},
}

TASK_DEFAULTS = {
    "column": None,
    "classes": None,
    "map": {},
    "only_where": None,
    "loss": {"type": "ce", "gamma": 2.0, "label_smoothing": 0.1},
    "class_weight_cap": 10.0,
    "balance_sampler": False,
    "group_labels": {"localizer": "drop", "phase": "inherit", "other_echo": "inherit"},
    "rules": {"enabled": False, "disable": [], "rename": {}},
    "train_labels": {"image": None, "meta": None},   # label sources per model; None = all
    "eval_labels": None,                              # label sources scored by evaluate.py; None = all
    "metric": "macro_f1",                             # or balanced_accuracy: paired tests, fusion weight
    "fusion_weight": 0.7,
    "threshold": 0.9,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    cfg["input"]["root"] = rel(cfg["input"]["root"])
    cfg["work_dir"] = rel(cfg["work_dir"])
    if not isinstance(cfg["labels"], dict):
        raise ValueError("labels must map a source name to a CSV, e.g. {expert: labels/expert.csv}")
    cfg["labels"] = {k: rel(p) for k, p in cfg["labels"].items()}
    every = list(cfg["labels"])
    tasks = {}
    for name, t in cfg["tasks"].items():
        if isinstance(t.get("rules"), bool):
            t = {**t, "rules": {"enabled": t["rules"]}}
        t = _merge(TASK_DEFAULTS, t)
        if not t["column"] or not t["classes"]:
            raise ValueError(f"task {name!r}: 'column' and 'classes' are required")
        t["train_labels"] = {m: (t["train_labels"].get(m) or every) for m in ("image", "meta")}
        t["eval_labels"] = t["eval_labels"] or every
        if t["metric"] not in ("macro_f1", "balanced_accuracy"):
            raise ValueError(f"task {name!r}: metric must be macro_f1 or balanced_accuracy")
        for src in t["train_labels"]["image"] + t["train_labels"]["meta"] + t["eval_labels"]:
            if src not in cfg["labels"]:
                raise ValueError(f"task {name!r}: label source {src!r} is not in labels {every}")
        if t["only_where"]:
            (parent, _), = t["only_where"].items()
            if parent not in tasks:
                raise ValueError(f"task {name!r}: only_where refers to {parent!r}, which must be defined before it")
        tasks[name] = t
    cfg["tasks"] = tasks
    return cfg


def paths(cfg: dict) -> dict:
    w = cfg["work_dir"]
    return {"index": os.path.join(w, "index.pkl"), "cache": os.path.join(w, "npy_cache"),
            "models": os.path.join(w, "models"), "eval": os.path.join(w, "evaluation")}
