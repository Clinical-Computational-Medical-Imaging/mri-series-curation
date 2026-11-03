#!/usr/bin/env python3
"""Train the image model and the metadata model of a task on all labelled series.

    python scripts/train.py --config config.yaml --task body
    python scripts/train.py --config config.yaml              # every task in the config

Each model trains on the label sources the task names in `train_labels`. Models go to
work_dir/models/<task>/. The fusion weight and the confidence threshold come from the config;
scripts/evaluate.py estimates both on your data.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curation.config import load_config, paths  # noqa: E402
from curation.dicom_index import Index  # noqa: E402
from curation.pipeline import model_dir, train_models, training_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, help="one task of the config (default: all)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    index = Index.load(paths(cfg)["index"], cfg["input"]["root"])
    for name in ([args.task] if args.task else list(cfg["tasks"])):
        info = train_models(cfg, name, index, training_rows(cfg, name, index), model_dir(cfg, name))
        logging.info("task %s trained: %s", name, {k: info[k] for k in
                     ("n_rows_image", "n_rows_meta", "n_patients", "best_val_f1", "temperature")})


if __name__ == "__main__":
    main()
