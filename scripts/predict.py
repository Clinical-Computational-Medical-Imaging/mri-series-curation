#!/usr/bin/env python3
"""Label every image group of the index with the trained models, rules and confidence gate.

    python scripts/predict.py --config config.yaml --out results/
    python scripts/predict.py --config config.yaml --out results/ --series-files nifti/series_files.csv

Tasks run in config order; a task with only_where (e.g. sequence only on BRAIN) labels the groups
whose parent label is that class, plus those where the parent was uncertain. Writes
  predictions.csv        one row per image group: key, series, study, group, localizer, and per
                         task: label, confidence, source (rule / model / model+mask), reason, uncertain
  review_<task>.csv      uncertain series (main group): fill the empty <column> with your label
                         and add the file as a label source, e.g. labels: {reviewed: review_<task>.csv}
With --series-files (scripts/convert.py): every labelled NIfTI is copied to
  <out>/nifti/<patient>/<study date>_<study>/<LABEL of the last task>_<file>.nii.gz
with its converter sidecar, enriched with plane, derived / complex flags and the labels of all tasks.
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curation.config import load_config, paths  # noqa: E402
from curation.dicom_index import Index  # noqa: E402
from curation.localizer import is_localizer  # noqa: E402
from curation.output import safe_name, write_sidecar  # noqa: E402
from curation.pipeline import label_keys, model_dir  # noqa: E402


def predict_all(cfg: dict, index) -> pd.DataFrame:
    keys = index.keys()
    df = pd.DataFrame({"key": keys})
    df["series_uid"] = [index.series_uid(k) for k in keys]
    df["study"] = [index.get(k)["study"] for k in keys]
    df["group"] = [index.get(k).get("group", "") for k in keys]
    df["is_main"] = [bool(index.get(k).get("is_main", True)) for k in keys]
    df["localizer"] = [is_localizer(index.get(k), cfg["localizer"]) or is_localizer(index.get(s), cfg["localizer"])
                       for k, s in zip(keys, df["series_uid"])]
    for name, task in cfg["tasks"].items():
        todo = ~df["localizer"]
        if task["only_where"]:
            (parent, value), = task["only_where"].items()
            todo &= (df[parent] == value) | (df[f"{parent}_uncertain"] == True)  # noqa: E712
        d = label_keys(cfg, name, index, df.loc[todo, "key"].tolist(), model_dir(cfg, name))
        d = d.rename(columns={c: (name if c == "label" else f"{name}_{c}") for c in d.columns if c != "key"})
        df = df.merge(d, on="key", how="left")
        logging.info("task %s: %d groups labelled, %d uncertain, sources %s", name, len(d),
                     int(d[f"{name}_uncertain"].sum()), d[f"{name}_source"].value_counts().to_dict())
    return df


def review_files(cfg: dict, index, df: pd.DataFrame, out: Path) -> None:
    pcol = cfg["patient_column"]
    for name, task in cfg["tasks"].items():
        u = df[df["is_main"] & (df[f"{name}_uncertain"] == True)]  # noqa: E712
        r = pd.DataFrame({"series_instance_uid": u["series_uid"],
                          pcol: [index.get(k).get("patient_id", "") for k in u["key"]],
                          "series_description": [index.get(k).get("series_description", "") for k in u["key"]],
                          f"{name}_predicted": u[name], f"{name}_confidence": u[f"{name}_confidence"].round(3),
                          task["column"]: ""})
        r.to_csv(out / f"review_{name}.csv", index=False)
        logging.info("review_%s.csv: %d series", name, len(r))


def copy_niftis(cfg: dict, index, df: pd.DataFrame, series_files: str, out: Path) -> None:
    tasks = list(cfg["tasks"])
    leaf = tasks[-1]
    sf = pd.read_csv(series_files, dtype=str)
    sf = sf[sf["side_output"].str.lower() != "true"]
    by_key = df.set_index("key")
    n, missing = 0, 0
    for r in sf.itertuples(index=False):
        keys = index.groups(r.series_instance_uid)
        key = (keys[0] if len(keys) == 1 else f"{r.series_instance_uid}~{r.group}") if keys else None
        if key not in by_key.index or pd.isna(by_key.at[key, leaf]):
            missing += 1
            continue
        row, e = by_key.loc[key], index.get(key)
        labels = {t: {"Label": row[t], "Confidence": round(float(row[f"{t}_confidence"]), 4),
                      "Source": row[f"{t}_source"], "Reason": row[f"{t}_reason"] or None,
                      "Uncertain": bool(row[f"{t}_uncertain"])} for t in tasks if not pd.isna(row[t])}
        dst = (out / "nifti" / safe_name(e.get("patient_id") or "unknown")
               / safe_name(f"{e.get('study_date', '')}_{Path(r.study).stem}") / f"{row[leaf]}_{Path(r.nifti_path).name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.nifti_path, dst)
        write_sidecar(Path(r.nifti_path), dst, labels)
        n += 1
    logging.info("NIfTI copied: %d; not labelled (localizer, other parent class, not in index): %d", n, missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--series-files", default=None, help="series_files.csv of scripts/convert.py")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    index = Index.load(paths(cfg)["index"], cfg["input"]["root"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = predict_all(cfg, index)
    df.to_csv(out / "predictions.csv", index=False)
    review_files(cfg, index, df, out)
    if args.series_files:
        copy_niftis(cfg, index, df, args.series_files, out)


if __name__ == "__main__":
    main()
