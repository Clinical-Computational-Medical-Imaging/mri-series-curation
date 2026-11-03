#!/usr/bin/env python3
"""Scan the DICOM input (config `input`) and write the index (work_dir/index.pkl).

    python scripts/build_index.py --config config.yaml
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curation.config import load_config, paths  # noqa: E402
from curation.dicom_index import Index  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    idx = Index(cfg["input"]["kind"], cfg["input"]["root"])
    counts = idx.build(cfg["workers"])
    os.makedirs(cfg["work_dir"], exist_ok=True)
    idx.save(paths(cfg)["index"])
    logging.info("index: %s", counts)


if __name__ == "__main__":
    main()
