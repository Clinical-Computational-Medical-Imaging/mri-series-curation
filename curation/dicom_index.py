"""DICOM index: one entry per IMAGE GROUP of every MR series.

ImageType and echo number are stored per image, and one series can hold several kinds of image:
magnitude + phase (e.g. Philips SWI), the PD and the T2 echo of a dual-echo series, a localizer
image inside an MPR series. Each series is therefore split into image groups (image_group: class
"m" / "loc" / "pha" x echo number). A series with one group keeps its entry under the series UID;
a mixed series gets one entry per group, "<uid>~m1", "<uid>~m2", "<uid>~pha1", ... Index.get(uid)
returns the series' main group (class "m", longest echo time: the T2 echo of PD+T2).

A series imported twice (in two studies) is kept once: the copy with most images, ties by study
name, so rebuilds are deterministic.

Nothing here prints UIDs or file names; warnings carry error types only.
"""
from __future__ import annotations

import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pydicom
from pydicom.config import settings as _dcm_settings
from tqdm import tqdm

from .sources import make_source

# DICOM quantities stored per group (first image of the group), in DICOM units
META_FIELDS = ["RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "MagneticFieldStrength",
               "NumberOfAverages", "SliceThickness", "PixelBandwidth", "SpacingBetweenSlices"]

_UID_JUNK = re.compile(r"(\.(0+|-\d+))+$")   # trailing junk like ".-1" or ".0" seen in some exports


def normalize_uid(uid: str) -> str:
    """Strip whitespace and non-standard trailing suffixes from a DICOM UID."""
    return _UID_JUNK.sub("", str(uid).strip())


def _safe_float(ds, attr: str) -> float:
    try:
        v = getattr(ds, attr, 0.0)
        if v is None:
            return 0.0
        if hasattr(v, "__iter__") and not isinstance(v, str):
            v = list(v)[0]
        return float(v)
    except Exception:
        return 0.0


def _tag_text(ds, attr: str) -> str:
    """A (possibly multi-valued) DICOM attribute as one string (['SE', 'IR'] -> 'SE IR')."""
    v = getattr(ds, attr, "") or ""
    if hasattr(v, "__iter__") and not isinstance(v, str):
        return " ".join(str(x) for x in v).strip()
    return str(v).strip()


# ── image groups ──────────────────────────────────────────────────────────────

_PHASE_PARTS = {"P", "PHASE", "I", "IMAGINARY"}
_PROTECTED_TOKENS = {"IN_PHASE", "OUT_PHASE", "OPP_PHASE"}   # Dixon magnitude images, not phase


def type_tokens(image_type) -> set[str]:
    """ImageType values as a token set; each value split on "_" (Philips SW_P -> SW, P) except the
    Dixon magnitude labels IN_PHASE / OUT_PHASE / OPP_PHASE, kept whole. dcm2niix writes those as
    two values ("IN", "PHASE"); adjacent IN / OUT / OPP + PHASE are re-joined first."""
    if isinstance(image_type, str):          # a single value comes as a string ("ORIGINAL\\PRIMARY")
        image_type = image_type.split("\\")
    vals = [v for v in (str(x).strip().upper() for x in (image_type or [])) if v]
    joined, i = [], 0
    while i < len(vals):
        if vals[i] in ("IN", "OUT", "OPP") and i + 1 < len(vals) and vals[i + 1] == "PHASE":
            joined.append(vals[i] + "_PHASE")
            i += 2
        else:
            joined.append(vals[i])
            i += 1
    out: set[str] = set()
    for t in joined:
        out.add(t)
        if t not in _PROTECTED_TOKENS:
            out.update(p for p in t.split("_") if p)
    return out


def _inversion_recovery(tokens: set, scanning_sequence, inversion_time) -> bool:
    """IR: ScanningSequence contains IR, an ImageType token is IR, or an InversionTime > 0."""
    try:
        ti = float(inversion_time[0] if isinstance(inversion_time, (list, tuple)) else inversion_time)
    except (TypeError, ValueError, IndexError):
        ti = 0.0
    return "IR" in str(scanning_sequence or "").upper() or "IR" in tokens or ti > 0


def image_group(image_type, echo_number=None, scanning_sequence="", inversion_time=None) -> tuple[str, int]:
    r"""Group of one image inside its series: (class, echo).

    class: "loc" localizer (any token contains LOCALIZER); "pha" phase / imaginary (P, PHASE, I,
    IMAGINARY) and the vendor's real-part token R outside inversion recovery (a real-reconstructed
    T1-IR is the diagnostic image); "m" everything else. A bare REAL (appended by dcm2niix, and by
    Philips on magnitude-type images) never makes an image phase. echo = int(EchoNumbers), missing
    -> 1. The same rule applies to a DICOM header and to a dcm2niix sidecar.

    >>> image_group(["ORIGINAL", "PRIMARY", "SW_P", "SW", "P"])[0], image_group(["ORIGINAL", "SW_M"])[0]
    ('pha', 'm')
    >>> image_group(["DERIVED", "PRIMARY", "DIXON", "OUT_PHASE"])[0], image_group("ORIGINAL\\IN_PHASE")[0]
    ('m', 'm')
    >>> image_group(["ORIGINAL", "PRIMARY", "M_IR", "R"], 2, "IR"), image_group(["ORIGINAL", "R"], None, "SE")
    (('m', 2), ('pha', 1))
    >>> image_group(["ORIGINAL", "R", "REAL"], None, "SE", 0.6)[0], image_group(["ORIGINAL", "ADC", "REAL"])[0]
    ('m', 'm')
    """
    it = type_tokens(image_type)
    if isinstance(scanning_sequence, (list, tuple)):
        scanning_sequence = "\\".join(str(x) for x in scanning_sequence)
    if any("LOCALIZER" in t for t in it):
        cls = "loc"
    elif it & _PHASE_PARTS or ("R" in it and not _inversion_recovery(it, scanning_sequence, inversion_time)):
        cls = "pha"
    else:
        cls = "m"
    try:
        echo = int(echo_number or 1)
    except (TypeError, ValueError):
        echo = 1
    return cls, echo


def group_class(key_or_group: str) -> str:
    """"m" / "loc" / "pha" of an index key ("<uid>~pha1") or group name ("pha1"); "m" for a plain key."""
    g = str(key_or_group).split("~", 1)[1] if "~" in str(key_or_group) else str(key_or_group)
    return g.rstrip("0123456789") or "m"


def _group_entries(uid: str, groups: dict) -> dict:
    """Index entries of one series: one group -> the entry under the series UID; several -> one
    entry per group under "<uid>~<class><echo>". Main group: longest-TE group of class "m" (ties:
    most images), else the largest group."""
    for g in groups.values():
        g["files"].sort(key=lambda x: x[0])
        g["n_slices_total"] = len(g["files"])
    if len(groups) == 1:
        (entry,) = groups.values()
        entry.update(series_uid=uid, group="", is_main=True)
        return {uid: entry}
    main_cands = [k for k in groups if k[0] == "m"] or list(groups)
    main = max(main_cands, key=lambda k: (groups[k]["echo_time"], groups[k]["n_slices_total"]))
    out = {}
    for (cls, echo), g in groups.items():
        g.update(series_uid=uid, group=f"{cls}{echo}", is_main=(cls, echo) == main)
        out[f"{uid}~{cls}{echo}"] = g
    return out


def _new_entry(ds, study: str, key: tuple[str, int]) -> dict:
    iop, ps = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "PixelSpacing", None)
    it = getattr(ds, "ImageType", None)
    return {
        "study": study,
        "files": [],
        "meta_vec": np.array([_safe_float(ds, f) for f in META_FIELDS], dtype=np.float32),
        "orientation": [float(x) for x in iop] if iop is not None and len(iop) >= 6 else [0.0] * 6,
        "pixel_spacing": [float(ps[0]), float(ps[1])] if ps is not None and len(ps) >= 2 else [0.0, 0.0],
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "series_description": str(getattr(ds, "SeriesDescription", "") or "").strip(),
        "scanning_sequence": _tag_text(ds, "ScanningSequence"),
        "sequence_variant": _tag_text(ds, "SequenceVariant"),
        "body_part_examined": _tag_text(ds, "BodyPartExamined"),
        "contrast_bolus_agent": str(getattr(ds, "ContrastBolusAgent", "") or "").strip(),
        "echo_number": key[1],
        "echo_time": _safe_float(ds, "EchoTime"),
        "image_type": (str(it).upper().split("\\") if isinstance(it, str)
                       else [str(x).strip().upper() for x in (it or [])]),
        "mr_acquisition_type": _tag_text(ds, "MRAcquisitionType"),
        "patient_id": str(getattr(ds, "PatientID", "") or "").strip(),
        "study_instance_uid": normalize_uid(getattr(ds, "StudyInstanceUID", "") or ""),
        "study_date": str(getattr(ds, "StudyDate", "") or "").strip(),
    }


def _index_one_study(kind: str, root: str, study: str) -> tuple[str, dict, list[str]]:
    """Scan one study (worker process): (study, {index key: entry}, warnings)."""
    try:
        _dcm_settings.reading_validation_mode = "IGNORE"
    except Exception:
        pass
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
    src = make_source(kind, root)
    series: dict = {}       # uid -> {(class, echo): entry}
    warns: list[str] = []
    try:
        with src.reader(study) as r:
            for name in src.members(study):     # every file: names / extensions do not matter
                try:
                    with r.open(name) as f:
                        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                except Exception as e:
                    warns.append(f"unreadable member: {type(e).__name__}")
                    continue
                if str(getattr(ds, "Modality", "MR")).upper() != "MR":
                    continue
                uid = normalize_uid(getattr(ds, "SeriesInstanceUID", "") or "")
                if not uid:
                    continue
                groups = series.setdefault(uid, {})
                key = image_group(getattr(ds, "ImageType", None), getattr(ds, "EchoNumbers", None),
                                  _tag_text(ds, "ScanningSequence"), getattr(ds, "InversionTime", None))
                if key not in groups:
                    groups[key] = _new_entry(ds, study, key)
                try:
                    inst = int(getattr(ds, "InstanceNumber", 0) or 0)
                except (TypeError, ValueError):
                    inst = 0
                groups[key]["files"].append((inst, name))
    except Exception as e:
        warns.append(f"study error: {type(e).__name__}")
    entries: dict = {}
    for uid, groups in series.items():
        entries.update(_group_entries(uid, groups))
    return study, entries, warns


class Index:
    """index key -> entry (study, files, meta_vec, header fields; see _new_entry)."""

    def __init__(self, kind: str, root: str):
        self.kind, self.root = kind, str(root)
        self.source = make_source(kind, root)
        self._data: dict = {}
        self._groups: dict = {}     # series uid -> [index keys]
        self._main: dict = {}       # series uid -> key of its main group

    # ── build / persist ──
    def build(self, workers: int = 8) -> dict:
        """Scan every study in parallel. Returns counts (studies, series, entries, duplicates, warnings)."""
        studies = self.source.studies()
        copies: dict = {}           # series uid -> {study: {key: entry}}
        n_warn: dict = {}
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(studies) or 1))) as pool:
            futs = [pool.submit(_index_one_study, self.kind, self.root, s) for s in studies]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Indexing studies"):
                study, entries, warns = fut.result()
                for key, e in entries.items():
                    copies.setdefault(e["series_uid"], {}).setdefault(study, {})[key] = e
                for w in warns:
                    n_warn[w] = n_warn.get(w, 0) + 1
        self._data, n_dup = {}, 0
        for by_study in copies.values():
            self._data.update(by_study[self._pick_copy(by_study)])
            n_dup += len(by_study) - 1
        self._link()
        return {"studies": len(studies), "series": len(self._groups), "entries": len(self._data),
                "duplicate_copies_dropped": n_dup, "warnings": n_warn}

    @staticmethod
    def _pick_copy(by_study: dict) -> str:
        """Study whose copy of a series is kept: most images over all its groups, ties -> smallest name."""
        return min(by_study, key=lambda s: (-sum(e["n_slices_total"] for e in by_study[s].values()), s))

    def _link(self) -> None:
        self._groups, self._main = {}, {}
        for key, e in self._data.items():
            uid = e.get("series_uid", key)
            self._groups.setdefault(uid, []).append(key)
            if e.get("is_main", True):
                self._main[uid] = key

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "root": self.root, "entries": self._data}, f)

    @classmethod
    def load(cls, path: str, root: Optional[str] = None) -> "Index":
        """root: override the stored input root (data moved)."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls(d["kind"], root or d["root"])
        idx._data = d["entries"]
        idx._link()
        return idx

    # ── lookup ──
    @staticmethod
    def _key(uid: str) -> str:
        uid = str(uid).strip()
        if "~" in uid:
            base, grp = uid.split("~", 1)
            return f"{normalize_uid(base)}~{grp}"
        return normalize_uid(uid)

    def get(self, uid: str) -> Optional[dict]:
        """Entry of an index key, or of a series UID its main group."""
        key = self._key(uid)
        return self._data.get(key) or self._data.get(self._main.get(key, ""))

    def groups(self, uid: str) -> list[str]:
        """Index keys of all image groups of a series."""
        return list(self._groups.get(self._key(uid), []))

    def series_uid(self, key: str) -> str:
        e = self._data.get(self._key(key))
        return e.get("series_uid", self._key(key)) if e else self._key(key).split("~", 1)[0]

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def series(self) -> list[str]:
        return list(self._groups.keys())

    def __len__(self) -> int:
        return len(self._data)
