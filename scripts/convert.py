#!/usr/bin/env python3
"""DICOM -> NIfTI with dcm2niix, one run per series, and a map of every NIfTI to its image group.

    python scripts/convert.py --config config.yaml --out /path/to/nifti

Writes <out>/<study>/<protocol>_<datetime>_<series number>[...].nii.gz + .json (dcm2niix -ba y:
anonymised BIDS sidecars, one per echo) and <out>/series_files.csv:
  study, series_instance_uid, nifti_path, group ("m1", "pha1", ... as in the index),
  side_output (dcm2niix extra output: _Eq, _Tilt, _ROI, _i<5 digits>, _ADC), main (one file per
  series and study: the main group's primary file with most voxels).
Compressed images (JPEG etc.) are decompressed with pydicom first where its handlers allow.
Logs counts and error categories only.
"""
import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curation.config import load_config  # noqa: E402
from curation.dicom_index import image_group, normalize_uid  # noqa: E402
from curation.sources import make_source  # noqa: E402

D2N_ARGS = ["-z", "y", "-b", "y", "-ba", "y", "-f", "%p_%t_%s", "-w", "2"]
SIDE = re.compile(r"_(Eq_\d+|Tilt_\d+|ROI\d+|i\d{5}|ADC)$")
FIELDS = ["study", "series_instance_uid", "nifti_path", "group", "side_output", "main"]


def _count(stats: dict, key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _convert_study(kind: str, root: str, study: str, out_root: str, dcm2niix: str) -> tuple[list[dict], dict]:
    import pydicom
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    src = make_source(kind, root)
    out = Path(out_root) / study.removesuffix(".zip")
    out.mkdir(parents=True, exist_ok=True)
    stats, rows = {}, []
    tmp = Path(tempfile.mkdtemp())
    try:
        series: dict[str, Path] = {}
        with src.reader(study) as r:
            for k, name in enumerate(src.members(study)):
                try:
                    with r.open(name) as f:
                        ds = pydicom.dcmread(f, force=True)
                    if str(getattr(ds, "Modality", "MR")).upper() != "MR" or "PixelData" not in ds:
                        continue
                    uid = normalize_uid(getattr(ds, "SeriesInstanceUID", "") or "")
                    if not uid:
                        continue
                    if ds.file_meta.TransferSyntaxUID.is_compressed:
                        try:
                            ds.decompress()
                        except Exception:
                            _count(stats, "not_decompressed")
                    d = series.setdefault(uid, tmp / f"s{len(series):05d}")
                    d.mkdir(exist_ok=True)
                    ds.save_as(d / f"{k:06d}.dcm")
                except Exception as e:
                    _count(stats, f"unreadable:{type(e).__name__}")
        for uid, d in series.items():
            before = {p.name for p in out.glob("*.nii.gz")}
            try:
                subprocess.run([dcm2niix, *D2N_ARGS, "-o", str(out), str(d)], capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                _count(stats, "timeout")
            new = sorted({p.name for p in out.glob("*.nii.gz")} - before)
            _count(stats, "series_ok" if new else "series_failed")
            rows.extend(_file_rows(study, uid, [out / n for n in new]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows, stats


def _file_rows(study: str, uid: str, files: list[Path]) -> list[dict]:
    import nibabel as nib
    import numpy as np
    rows = []
    for f in files:
        stem = f.name[:-7]
        js = f.with_name(stem + ".json")
        try:
            sc = json.loads(js.read_text()) if js.exists() else {}
        except ValueError:
            sc = {}
        cls, echo = image_group(sc.get("ImageType"), sc.get("EchoNumber"), sc.get("ScanningSequence"),
                                sc.get("InversionTime"))
        try:
            shape = nib.load(str(f)).header.get_data_shape()
            n_img, n_vox = (int(np.prod(shape[2:])) if len(shape) > 2 else 1), int(np.prod(shape))
        except Exception:
            n_img = n_vox = 0
        rows.append({"study": study, "series_instance_uid": uid, "nifti_path": str(f), "group": f"{cls}{echo}",
                     "side_output": bool(SIDE.search(stem)), "main": False,
                     "_te": float(sc.get("EchoTime") or -1), "_n_img": n_img, "_n_vox": n_vox})
    if rows:   # main file: main group (class "m", longest TE, ties most images), its largest primary file
        groups: dict = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        n_img = {g: sum(r["_n_img"] for r in v) for g, v in groups.items()}
        m = [g for g in groups if g.startswith("m")]
        main = (max(m, key=lambda g: (max(r["_te"] for r in groups[g]), n_img[g], g)) if m
                else max(groups, key=lambda g: (n_img[g], g)))
        cand = [r for r in groups[main] if not r["side_output"]] or groups[main]
        max(cand, key=lambda r: (r["_n_vox"], r["_n_img"], r["nifti_path"]))["main"] = True
    return [{k: r[k] for k in FIELDS} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dcm2niix", default=shutil.which("dcm2niix") or "dcm2niix")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    src = make_source(cfg["input"]["kind"], cfg["input"]["root"])
    os.makedirs(args.out, exist_ok=True)
    rows, stats = [], {}
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futs = [pool.submit(_convert_study, cfg["input"]["kind"], cfg["input"]["root"], s, args.out, args.dcm2niix)
                for s in src.studies()]
        for fut in as_completed(futs):
            r, st = fut.result()
            rows.extend(r)
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
    with open(os.path.join(args.out, "series_files.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logging.info("converted: %d files; %s", len(rows), stats)


if __name__ == "__main__":
    main()
