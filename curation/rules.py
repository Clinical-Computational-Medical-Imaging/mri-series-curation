"""Header rules from MR physics and DICOM conventions, applied after the models.

  identifying rules  the header says what a series IS          -> one class, overrides the models
  excluding rules    the header says what a series CANNOT be   -> classes masked out of the models

12 identifying + 5 excluding rules (RULES below). Physics / standard rules (ImageType, inversion
timing, echo / repetition times) fire only when the field is present; name rules (description)
depend on what was typed at the scanner. A missing value never triggers a rule.

The target classes are those of the example sequence task (T1 T1C T2 FLAIR DWI ADC SWI T2STAR
OTHER), with its conventions: STIR, proton density, bSSFP/CISS-type, phase images, synthetic MR,
exponential ADC and perfusion are OTHER. For another class space, rename targets in the config
(`rules.rename`, e.g. {OTHER: MISC}) or disable rules (`rules.disable`); a rule whose target class
is not in the task is skipped with a warning, and excluded classes not in the task are ignored.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .dicom_index import META_FIELDS, image_group
from .features import image_type, meta_value

log = logging.getLogger(__name__)

# "\b" does not separate at "_" (t2w_drive, t2_stir), so the boundaries are explicit non-letters
NAME_BSSFP = r"ciss|fiesta|trufi|true[ _-]?fisp|b[_-]?ffe|(^|[^a-z])drive([^a-z]|$)|cosmic|(^|[^a-z])ssfp"
NAME_STIR_ID = r"(^|[^a-z])stir([^a-z]|$)"
NAME_FLAIR = r"flair|dark[ _-]?fl|fluid"
NAME_DIR = r"(^|[^a-z])dir([^a-z]|$)|double[ _-]?(ir|inv)"
NAME_SYNTHETIC = r"magic|synthetic|symri"
NAME_PERFUSION = r"perf|(^|[^a-z])(dsc|dce|pwi|r?cbv|r?cbf|mtt|ttp|tmax)([^a-z]|$)|bolus"


def _lt(v, x):
    return v is not None and v < x


def _gt(v, x):
    return v is not None and v > x


def _btw(v, a, b):
    return v is not None and a <= v <= b


class Header:
    """Rule inputs, times in ms. From a dcm2niix sidecar (times in s) or an index entry (from_entry)."""

    def __init__(self, sidecar: dict | None, description: str = "", echo_times=None, nifti_name: str = ""):
        sc = sidecar or {}
        self.name = str(nifti_name or "")
        self.echo_times = [t * 1000.0 for t in (echo_times or [])]     # every echo of the series
        it = sc.get("ImageType") or []
        it = it.split("\\") if isinstance(it, str) else it
        self.it = [str(x).strip().upper() for x in it]
        self.ss = str(sc.get("ScanningSequence", "") or "").upper()
        ti = sc.get("InversionTime")
        self.n_ti = len(ti) if isinstance(ti, (list, tuple)) else int(ti is not None)
        if isinstance(ti, (list, tuple)):
            ti = ti[0] if ti else None
        self.te, self.tr, self.ti = self._ms(sc.get("EchoTime")), self._ms(sc.get("RepetitionTime")), self._ms(ti)
        self.variant = " ".join(str(sc.get(k, "") or "") for k in
                                ("SequenceVariant", "ScanOptions", "SequenceName", "PulseSequenceName")).lower()
        self.fa = self._num(sc.get("FlipAngle"))
        self.desc = (description or str(sc.get("SeriesDescription", "") or "")).lower()
        self.se, self.gr, self.ir, self.ep = ("SE" in self.ss, "GR" in self.ss, "IR" in self.ss, "EP" in self.ss)

    @classmethod
    def from_entry(cls, entry: dict, index=None) -> "Header":
        """From an index entry (DICOM units, ms) and the echo times of all groups of its series."""
        raw = {n: meta_value(entry, n) for n in META_FIELDS}
        te = entry.get("echo_time")
        te = float(te) if te not in (None, "") else (raw["EchoTime"] or None)
        sec = lambda v: None if not v else v / 1000.0
        sidecar = {"ImageType": image_type(entry), "ScanningSequence": str(entry.get("scanning_sequence", "")),
                   "SequenceVariant": str(entry.get("sequence_variant", "")),
                   "EchoTime": sec(te), "RepetitionTime": sec(raw["RepetitionTime"]),
                   "InversionTime": sec(raw["InversionTime"]), "FlipAngle": raw["FlipAngle"] or None}
        own = [float(entry["echo_time"])] if entry.get("echo_time") not in (None, "") else []
        tes = own
        if index is not None and entry.get("series_uid"):
            tes = sorted({float(e["echo_time"]) for k in index.groups(entry["series_uid"])
                          if (e := index.get(k)) and e.get("echo_time") not in (None, "")}) or own
        return cls(sidecar, str(entry.get("series_description", "") or ""), [t / 1000.0 for t in tes])

    @staticmethod
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _ms(cls, v):
        v = cls._num(v)
        return None if v is None else v * 1000.0

    def has(self, token: str) -> bool:
        return any(token in x for x in self.it)

    @property
    def spectroscopy(self) -> bool:
        """Spectroscopy and its reference images: header times belong to the spectroscopy sequence."""
        return any(t in x for x in self.it for t in ("SPECTROSCOPY", "CSI", "SVS")) or bool(
            re.search(r"\bsvs|\bcsi\b|spectro|\bpress\b|\bsteam\b|\bmrs\b", self.desc))

    @property
    def diffusion(self) -> bool:
        return self.ep or any(t in x for x in self.it for t in ("DIFFUSION", "ADC", "TRACE")) or bool(
            re.search(r"dwi|diff|adc|trace|(^|[^a-z0-9])b ?\d{3,4}", self.desc))

    @property
    def exp_adc(self) -> bool:
        """Exponential ADC: looks like DWI, is computed."""
        return (any(x in ("EXP", "EADC") or x.startswith("EXP_") for x in self.it)
                or bool(re.search(r"(^|[^a-z])eadc([^a-z]|$)|exp(onential)?[ _]?adc", self.desc))
                or (self.diffusion and bool(re.search(r"(^|[^a-z])exp([^a-z]|$)", self.desc))))

    @property
    def complex_part(self) -> bool:
        """Phase / imaginary images, and real-part images except of inversion recovery (same rule as the image groups)."""
        return image_group(self.it, None, self.ss, None if self.ti is None else self.ti / 1000.0)[0] == "pha"

    @property
    def dcm2niix_adc(self) -> bool:
        """dcm2niix's _ADC output: the ADC map of a DWI series, which inherits the DWI sidecar."""
        if not self.name:
            return False
        stem = Path(self.name).name
        stem = stem[:-7] if stem.endswith(".nii.gz") else stem.removesuffix(".nii")
        return stem.endswith("_ADC")

    @property
    def synthetic(self) -> bool:
        return self.has("SYNTHETIC") or bool(re.search(NAME_SYNTHETIC, self.desc))

    @property
    def double_ir(self) -> bool:
        return self.n_ti > 1 or bool(re.search(NAME_DIR, self.desc)) or bool(re.search(r"(^|[^a-z])dir", self.variant))

    @property
    def proton_density(self) -> bool:
        """PD-weighted spin echo: no IR / EPI / gradient echo, and TE < 40 ms with TR > 1500 ms, or the
        short echo (TE < 60 ms) of a multi-echo SE series that also has an echo >= 60 ms (PD+T2)."""
        if not self.se or self.ir or self.ep or self.gr or self.te is None:
            return False
        if self.te < 40 and _gt(self.tr, 1500):
            return True
        return len(self.echo_times) > 1 and self.te < 60 and max(self.echo_times) >= 60

    @property
    def perfusion(self) -> bool:
        return self.has("PERFUSION") or bool(re.search(NAME_PERFUSION, self.desc))


# (rule id, target class, reason, test); evaluated in this order, the first that fires decides
IDENTIFY = [
    ("phase", "OTHER", "phase / real / imaginary image (ImageType)", lambda h: h.complex_part),
    ("dcm2niix_adc", "ADC", "dcm2niix _ADC output (ADC map of the DWI series)", lambda h: h.dcm2niix_adc),
    ("synthetic", "OTHER", "synthetic MR (MAGiC / SyMRI)", lambda h: h.synthetic),
    ("exp_adc", "OTHER", "exponential ADC: derived diffusion map", lambda h: h.exp_adc),
    ("imagetype_adc", "ADC", "ImageType ADC", lambda h: any(x == "ADC" or x.startswith("ADC_") for x in h.it)),
    ("tracew", "DWI", "ImageType TRACEW (diffusion trace)", lambda h: h.has("TRACEW")),
    ("flair_ir", "FLAIR", "IR, TI > 1500 ms, TE > 60 ms",
     lambda h: h.ir and _gt(h.ti, 1500) and _gt(h.te, 60) and not h.double_ir),
    ("stir_ti", "OTHER", "IR, TI 100-300 ms (STIR)", lambda h: h.ir and _btw(h.ti, 100, 300)),
    ("stir_name", "OTHER", "description names STIR",
     lambda h: bool(re.search(NAME_STIR_ID, h.desc)) and not re.search(NAME_FLAIR, h.desc)
     and not _gt(h.ti, 400) and not h.double_ir),
    ("bssfp_name", "OTHER", "description names bSSFP / CISS-type", lambda h: bool(re.search(NAME_BSSFP, h.desc))),
    ("proton_density", "OTHER", "proton density: spin echo, TE < 40 ms and TR > 1500 ms, or short echo of PD+T2",
     lambda h: h.proton_density),
    ("perfusion", "OTHER", "perfusion (DSC / DCE) or perfusion map", lambda h: h.perfusion),
]

# (rule id, excluded classes, reason, test); the first that fires decides
EXCLUDE = [
    ("t1_ir", {"T2", "FLAIR", "DWI", "ADC", "SWI", "T2STAR"}, "IR, TI 600-1200 ms, TE < 30 ms (T1-weighted)",
     lambda h: h.ir and _btw(h.ti, 600, 1200) and _lt(h.te, 30)),
    ("se_short_te", {"T2", "FLAIR"}, "spin echo, TE < 60 ms (not T2-weighted)",
     lambda h: h.se and not h.ir and not h.ep and _lt(h.te, 60)),
    ("gre_short_te", {"T2", "FLAIR"}, "gradient echo, TE < 40 ms (not T2-TSE)",
     lambda h: h.gr and not h.se and _lt(h.te, 40)),
    ("t2_weighted", {"T1", "T1C"}, "TE > 60 ms and TR > 2000 ms (T2-weighted)",
     lambda h: _gt(h.te, 60) and _gt(h.tr, 2000) and not h.ir),
    ("not_flair", {"FLAIR"}, "TE < 60 ms without long-TI inversion (not FLAIR)",
     lambda h: _lt(h.te, 60) and not h.ir),
]


class RuleSet:
    """The rules of one task (config `rules`: enabled, disable, rename) for its class list."""

    def __init__(self, conf: dict, classes: list[str]):
        self.enabled = bool(conf.get("enabled", False))
        rename, off = conf.get("rename", {}) or {}, set(conf.get("disable", []) or [])
        unknown = off - {r[0] for r in IDENTIFY + EXCLUDE}
        if unknown:
            raise ValueError(f"rules.disable: unknown rule ids {sorted(unknown)}")
        self.identify_rules, self.exclude_rules = [], []
        for rid, target, why, test in IDENTIFY:
            target = rename.get(target, target)
            if rid in off:
                continue
            if target not in classes:
                if self.enabled:
                    log.warning("rule %s skipped: target class %s is not in the task", rid, target)
                continue
            self.identify_rules.append((rid, target, why, test))
        for rid, banned, why, test in EXCLUDE:
            if rid not in off:
                self.exclude_rules.append((rid, {rename.get(c, c) for c in banned} & set(classes), why, test))

    def identify(self, h: Header):
        """(rule id, class, reason) of the first identifying rule that fires, else None."""
        if not self.enabled or h.spectroscopy:
            return None
        return next(((rid, c, why) for rid, c, why, test in self.identify_rules if test(h)), None)

    def exclude(self, h: Header) -> tuple[set, str]:
        if not self.enabled:
            return set(), ""
        return next(((b, why) for rid, b, why, test in self.exclude_rules if test(h)), (set(), ""))

    def apply(self, probs, classes: list[str], h: Header) -> tuple[str, float, str, str]:
        """Final (label, confidence, source, reason) of one image group from its fused probabilities:
        identifying rule -> its class, confidence 1; else the most probable class not excluded,
        confidence renormalised over the remaining classes."""
        import numpy as np
        ident = self.identify(h)
        if ident:
            return ident[1], 1.0, "rule", ident[2]
        banned, why = self.exclude(h)
        p = np.asarray(probs, float)
        keep = [i for i, c in enumerate(classes) if c not in banned] or list(range(len(classes)))
        i = max(keep, key=lambda j: p[j])
        conf = float(p[i] / p[keep].sum()) if banned and p[keep].sum() > 0 else float(p[i])
        return classes[i], conf, ("model+mask" if banned else "model"), why
