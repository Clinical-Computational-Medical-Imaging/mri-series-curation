"""Labels: CSV files with one row per series, turned into one training row per image group.

Label CSV format. Any number of named sources (config `labels`, e.g. expert and reviewed); each
task chooses per model which sources it trains on (`train_labels`) and which are scored
(`eval_labels`). Merged sources: later ones win per series and column, so a file of reviewed
labels can correct an earlier one.

    series_instance_uid,patient_id,body_part,sequence_type
    1.2.3...,P001,BRAIN,T1C

Only series_instance_uid is required; the patient column (config `patient_column`) keeps all
series of a patient on one side of every split (without it, the study is used). One column per
task; empty or SKIP = not labelled. Values are mapped with the task's `map` (e.g. SPINE -> REST)
and must then be one of the task's classes.

Labels are per series; image groups get theirs from the task's `group_labels` policy:
  localizer / phase / other_echo:  inherit | drop | <class> | {when_label: T2, te_below: 60, label: OTHER}
The main group always carries the series label.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .dicom_index import group_class, normalize_uid
from .localizer import is_localizer

log = logging.getLogger(__name__)


def patient_key(patient, study_uid=None, series_uid=None) -> str:
    """The patient column if set; else a hashed placeholder of the study (all its series belong to
    one patient), else of the series. Hashed, so no UID ends up in fold files or logs."""
    if isinstance(patient, str) and patient.strip() and patient.strip().lower() != "nan":
        return patient.strip()
    for prefix, uid in (("study", study_uid), ("series", series_uid)):
        if isinstance(uid, str) and uid.strip() and uid.strip().lower() != "nan":
            return f"nopatient_{prefix}_" + hashlib.md5(normalize_uid(uid).encode()).hexdigest()[:10]
    return "nopatient_unknown"


def read_labels(cfg: dict, sources: list[str] | None = None) -> pd.DataFrame:
    """Label files of the given sources (config `labels` names; default all) merged into one row per
    series, later sources winning cell by cell."""
    merged = None
    for path in [cfg["labels"][s] for s in (sources or list(cfg["labels"]))]:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["series_instance_uid"] = df["series_instance_uid"].astype(str).map(normalize_uid)
        df = df.replace({"": np.nan}).drop_duplicates("series_instance_uid", keep="last").set_index("series_instance_uid")
        merged = df if merged is None else df.combine_first(merged)
    if merged is None:
        raise ValueError("no label files given (config: labels)")
    return merged.reset_index()


def _class_of(raw, task: dict):
    if not isinstance(raw, str) or not raw.strip():
        return None
    v = task["map"].get(raw.strip(), raw.strip())
    return v if v in task["classes"] else None


def _group_label(series_label: str, entry: dict, policy: dict):
    """Label of one image group from its series label (None = not trained)."""
    if entry.get("is_main", True):
        return series_label
    cls = group_class(entry.get("group", "m"))
    rule = policy.get({"loc": "localizer", "pha": "phase"}.get(cls, "other_echo"), "inherit")
    if isinstance(rule, dict):
        hit = (series_label == rule.get("when_label") and
               float(entry.get("echo_time") or 0) < float(rule.get("te_below", 0)))
        return rule["label"] if hit else series_label
    return {"inherit": series_label, "drop": None}.get(rule, rule)


def task_rows(cfg: dict, task_name: str, index, labels: pd.DataFrame) -> pd.DataFrame:
    """Training rows of a task, one per image group: key, series_uid, patient, label.
    Leaves out unlabelled series, labels outside the classes, series not in the index, localizer
    series, rows outside the task's only_where condition and groups the policy drops."""
    task = cfg["tasks"][task_name]
    col = task["column"]
    df = labels[labels[col].notna()].copy() if col in labels.columns else labels.iloc[:0].copy()
    counts = {"labelled": len(df)}
    df["label"] = df[col].map(lambda v: _class_of(v, task))
    skip = df[col].astype(str).str.strip().eq("SKIP")
    counts["not_a_class"] = int((df["label"].isna() & ~skip).sum())
    df = df[df["label"].notna()]
    if task["only_where"]:
        (parent, value), = task["only_where"].items()
        ptask = cfg["tasks"][parent]
        if ptask["column"] not in df.columns:
            raise ValueError(f"task {task_name!r}: only_where needs the column {ptask['column']!r} in the labels")
        df = df[df[ptask["column"]].map(lambda v: _class_of(v, ptask)) == value]
        counts["only_where"] = len(df)
    in_index = df["series_instance_uid"].map(lambda u: index.get(u) is not None)
    counts["not_in_index"] = int((~in_index).sum())
    df = df[in_index]
    loc = df["series_instance_uid"].map(lambda u: is_localizer(index.get(u), cfg["localizer"]))
    counts["localizer"] = int(loc.sum())
    df = df[~loc]
    pcol = cfg["patient_column"]
    rows = []
    for r in df.itertuples(index=False):
        uid = r.series_instance_uid
        patient = patient_key(getattr(r, pcol, None) if pcol in df.columns else None,
                              index.get(uid).get("study_instance_uid"), uid)
        for key in index.groups(uid):
            lab = _group_label(r.label, index.get(key), task["group_labels"])
            if lab is not None and lab in task["classes"]:
                rows.append({"key": key, "series_uid": uid, "patient": patient, "label": lab,
                             "raw_label": getattr(r, col)})
    out = pd.DataFrame(rows, columns=["key", "series_uid", "patient", "label", "raw_label"])
    counts.update(rows=len(out), series=out["series_uid"].nunique(), patients=out["patient"].nunique())
    log.info("task %s: %s", task_name, counts)
    return out


def patient_folds(patients, n_folds: int = 5, seed: int = 42) -> list[set]:
    """Patient-level folds: seeded shuffle, strided assignment (balanced fold sizes)."""
    import random
    p = sorted(set(patients))
    random.Random(seed).shuffle(p)
    return [set(p[i::n_folds]) for i in range(n_folds)]


def patient_split(patients, val_fraction: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
    """Row indices (train, validation), validation = a seeded fraction of the patients."""
    import random
    p = sorted(set(patients))
    random.Random(seed).shuffle(p)
    val = set(p[:max(1, int(len(p) * val_fraction))])
    return ([i for i, x in enumerate(patients) if x not in val], [i for i, x in enumerate(patients) if x in val])
