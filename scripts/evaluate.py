#!/usr/bin/env python3
"""Patient-level cross-validation of a task, and optionally one score on a held-out test set.

    python scripts/evaluate.py --config config.yaml --task sequence --folds 5
    python scripts/evaluate.py --config config.yaml --task sequence --test-patients test_patients.txt

Every patient of the evaluation labels (task `eval_labels`) is predicted once by models that did
not see it (pooled out-of-fold predictions); each fold trains both models on the other patients
with their own label sources (`train_labels`). Reported, as macro-F1 and balanced accuracy with
patient-bootstrap 95 % CIs: image model alone, metadata model alone, the fusion with and without
the header rules; with the task's `metric`: paired differences and the fusion weight with the best
out-of-fold score; and the confidence thresholds for error budgets (point estimate and the safer
bootstrap upper bound).
--test-patients (one patient id per line): these patients are left out of the CV and of every
training; after the CV the models are trained on all other patients and the test set is scored
once with the configured fusion weight and the CV's thresholds.
Writes work_dir/evaluation/<task>/{oof.csv, test.csv, report.json}; prints scores only.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curation import stats  # noqa: E402
from curation.config import load_config, paths  # noqa: E402
from curation.dicom_index import Index  # noqa: E402
from curation.fusion import choose_weight, decide, fuse  # noqa: E402
from curation.labels import patient_folds, read_labels, task_rows  # noqa: E402
from curation.pipeline import predict_probs, train_models, training_rows  # noqa: E402
from curation.rules import RuleSet  # noqa: E402


def score(y, pred, patients) -> dict:
    out = {}
    for m in stats.METRICS:
        v, lo, hi = stats.bootstrap(y, pred, patients, m)
        out[m] = round(v, 4)
        out[f"{m}_ci"] = [round(lo, 4), round(hi, 4)]
    out["accuracy"] = round(float(np.mean(np.asarray(y) == np.asarray(pred))), 4)
    return out


def analyse(ev, p_img, p_meta, classes, rules, index, weight, budgets, metric, thresholds=None) -> dict:
    y, pats, keys = ev["label"].tolist(), ev["patient"].to_numpy(), ev["key"].tolist()

    def paired(a, b) -> dict:
        d, lo, hi, p = stats.paired_bootstrap(y, a, b, pats, metric)
        return {"metric": metric, "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)], "p": round(p, 4)}

    image = np.array(classes)[np.nan_to_num(p_img, nan=0).argmax(1)]
    meta = np.array(classes)[p_meta.argmax(1)]
    fused = fuse(p_img, p_meta, weight)
    plain = np.array(classes)[fused.argmax(1)]
    final = decide(keys, fused, classes, rules, index)
    rep = {"n_groups": len(ev), "n_patients": int(ev["patient"].nunique()), "fusion_weight": weight,
           "image": score(y, image, pats), "meta": score(y, meta, pats),
           "fused_without_rules": score(y, plain, pats), "fused": score(y, final["label"], pats),
           "fused_minus_image": paired(final["label"], image),
           "fused_minus_meta": paired(final["label"], meta),
           "rules_effect": paired(final["label"], plain),
           "decided_by_rule": round(float((final["source"] == "rule").mean()), 4)}
    if thresholds is None:
        thresholds = {f"{b:g}": {"plain": stats.threshold_for_budget(y, final["label"], final["confidence"], pats, b, safe=False),
                                 "safe": stats.threshold_for_budget(y, final["label"], final["confidence"], pats, b, safe=True)}
                      for b in budgets}
    rep["thresholds"] = {b: {kind: {"threshold": t, **dict(zip(("coverage", "error"),
                                                               map(lambda v: round(v, 4), stats.coverage(y, final["label"], final["confidence"], t))))}
                             for kind, t in th.items()} for b, th in thresholds.items()}
    return rep, final, {b: {k: v for k, v in th.items()} for b, th in thresholds.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-patients", default=None)
    ap.add_argument("--budgets", default="0.03,0.02", help="error budgets for the thresholds")
    ap.add_argument("--epochs", type=int, default=None, help="override image_model.epochs (quick runs)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.epochs:
        cfg["image_model"]["epochs"] = args.epochs
    task = cfg["tasks"][args.task]
    classes = list(task["classes"])
    rules = RuleSet(task["rules"], classes)
    budgets = [float(b) for b in args.budgets.split(",")]
    index = Index.load(paths(cfg)["index"], cfg["input"]["root"])
    out = os.path.join(paths(cfg)["eval"], args.task)
    os.makedirs(out, exist_ok=True)

    ev = task_rows(cfg, args.task, index, read_labels(cfg, task["eval_labels"]))
    test = set()
    if args.test_patients:
        with open(args.test_patients) as f:
            test = {line.strip() for line in f if line.strip()}
    dev = ev[~ev["patient"].isin(test)].reset_index(drop=True)
    p_img = np.full((len(dev), len(classes)), np.nan)
    p_meta = np.zeros((len(dev), len(classes)))
    for i, fold in enumerate(patient_folds(dev["patient"], args.folds, cfg["seed"])):
        rows = training_rows(cfg, args.task, index, exclude_patients=fold | test)
        fdir = os.path.join(out, f"fold{i}")
        train_models(cfg, args.task, index, rows, fdir)
        sel = dev["patient"].isin(fold).to_numpy()
        p_img[sel], p_meta[sel] = predict_probs(cfg, index, dev.loc[sel, "key"].tolist(), fdir)
        logging.info("fold %d done: %d groups", i, int(sel.sum()))

    metric = task["metric"]
    best_w, grid = choose_weight(dev["label"].tolist(), p_img, p_meta, classes, rules, dev["key"].tolist(), index, metric)
    report = {"task": args.task, "folds": args.folds, "metric": metric,
              "weight_grid": {f"{w:g}": round(s, 4) for w, s in grid.items()}, "best_weight": best_w}
    report["cv"], final, thresholds = analyse(dev, p_img, p_meta, classes, rules, index, task["fusion_weight"], budgets, metric)
    pd.concat([dev[["key", "label"]], final.drop(columns="key")], axis=1).to_csv(os.path.join(out, "oof.csv"), index=False)

    if test:
        te = ev[ev["patient"].isin(test)].reset_index(drop=True)
        rows = training_rows(cfg, args.task, index, exclude_patients=test)
        fdir = os.path.join(out, "all_development")
        train_models(cfg, args.task, index, rows, fdir)
        ti, tm = predict_probs(cfg, index, te["key"].tolist(), fdir)
        report["test"], tfinal, _ = analyse(te, ti, tm, classes, rules, index, task["fusion_weight"], budgets, metric,
                                            thresholds)
        pd.concat([te[["key", "label"]], tfinal.drop(columns="key")], axis=1).to_csv(os.path.join(out, "test.csv"), index=False)

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
