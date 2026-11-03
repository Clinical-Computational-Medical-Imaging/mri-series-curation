"""Output: each labelled NIfTI with the converter's JSON sidecar, enriched with

  ImagingPlane / ImagingPlaneObliquityDeg / ImagingPlaneIsOblique / ImagingPlaneSource
      from ImageOrientationPatientDICOM (slice normal), the NIfTI affine if that is missing
  IsDerived / ComplexComponent / Derivation      from ImageType
  CurationLabels: {task: {Label, Confidence, Source, Reason, Uncertain}}

UIDs and patient fields are dropped from the sidecar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .dicom_index import image_group, type_tokens

AXIS = ["Sagittal", "Coronal", "Axial"]
OBLIQUE_DEG = 30.0
DERIV_TOKENS = ("MPR", "ADC", "DIFFUSION", "PERFUSION", "PROJECTION IMAGE", "REFORMATTED", "MIP", "MNIP",
                "CPR", "PROCESSED", "TRACEW", "FA", "EXP")
DROP = {"SeriesInstanceUID", "StudyInstanceUID", "PatientName", "PatientID", "PatientBirthDate"}


def _stem(name: str) -> str:
    name = Path(str(name)).name
    return name[:-7] if name.endswith(".nii.gz") else name.removesuffix(".nii")


def plane_of(normal) -> tuple[str | None, float | None]:
    n = np.asarray(normal, dtype=float)
    ln = np.linalg.norm(n)
    if ln < 1e-6:
        return None, None
    n = n / ln
    i = int(np.argmax(np.abs(n)))
    return AXIS[i], float(np.degrees(np.arccos(min(abs(n[i]), 1.0))))


def plane(sidecar: dict, nifti: Path | None) -> tuple[str | None, float | None, str | None]:
    iop = sidecar.get("ImageOrientationPatientDICOM")
    if iop and len(iop) == 6:
        v = [float(x) for x in iop]
        p, a = plane_of(np.cross(v[:3], v[3:]))
        if p:
            return p, a, "dicom"
    if nifti is not None:
        try:
            import nibabel as nib
            aff = np.asarray(nib.load(str(nifti)).affine)[:3, :3]
            p, a = plane_of(aff[:, int(np.argmax(np.linalg.norm(aff, axis=0)))])
            return p, a, "nifti_affine"
        except Exception:
            pass
    return None, None, None


def image_type_info(tokens, scanning_sequence="", inversion_time=None) -> tuple[bool | None, str | None, str | None]:
    """(IsDerived, ComplexComponent M/P/R/I, Derivation) from ImageType."""
    if not tokens:
        return None, None, None
    tokens = tokens.split("\\") if isinstance(tokens, str) else tokens
    up = [str(t).strip().upper() for t in tokens]
    tok = type_tokens(up)
    if tok & {"P", "PHASE"}:
        comp = "P"
    elif tok & {"I", "IMAGINARY"}:
        comp = "I"
    elif image_group(up, None, scanning_sequence, inversion_time)[0] == "pha" or ("R" in tok and not tok & {"M", "MAGNITUDE"}):
        comp = "R"
    elif tok & {"M", "MAGNITUDE"}:
        comp = "M"
    else:
        comp = None
    return "DERIVED" in up, comp, next((t for t in up if t.startswith(DERIV_TOKENS)), None)


def enrich(src: dict, nifti: Path | None = None, labels: dict | None = None) -> dict:
    sc = {k: v for k, v in src.items() if k not in DROP}
    p, angle, source = plane(src, nifti)
    derived, comp, deriv = image_type_info(src.get("ImageType"), src.get("ScanningSequence"), src.get("InversionTime"))
    extra = {"ImagingPlane": p,
             "ImagingPlaneObliquityDeg": round(angle, 2) if angle is not None else None,
             "ImagingPlaneIsOblique": (angle >= OBLIQUE_DEG) if angle is not None else None,
             "ImagingPlaneSource": source, "IsDerived": derived, "ComplexComponent": comp, "Derivation": deriv}
    sc.update({k: v for k, v in extra.items() if v is not None})
    if labels:
        sc["CurationLabels"] = labels
    return sc


def source_sidecar(nifti: Path) -> Path | None:
    """The converter's sidecar of a NIfTI. dcm2niix writes extra outputs (_ADC, _Eq_1, ...) without
    a sidecar of their own: strip trailing name parts until one exists."""
    parts = _stem(nifti.name).split("_")
    for k in range(len(parts), 1, -1):
        j = nifti.with_name("_".join(parts[:k]) + ".json")
        if j.exists():
            return j
    return None


def write_sidecar(src_nifti: Path, dst_nifti: Path, labels: dict) -> bool:
    src = source_sidecar(Path(src_nifti))
    if src is None:
        return False
    try:
        base = json.loads(src.read_text())
    except ValueError:
        base = {"SidecarError": "converter sidecar unreadable (invalid JSON)"}
    sc = enrich(base, dst_nifti if Path(dst_nifti).exists() else Path(src_nifti), labels)
    if src.stem != _stem(Path(src_nifti).name):
        sc["SidecarFrom"] = "series base file (dcm2niix extra output)"
        if _stem(src_nifti).endswith("_ADC"):
            sc.update(Derivation="dcm2niix_ADC", IsDerived=True, ComplexComponent="M")
    out = Path(dst_nifti)
    out.with_name(_stem(out.name) + ".json").write_text(json.dumps(sc, indent=2))
    return True


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unknown"
