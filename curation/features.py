"""Metadata input of the metadata model, from an index entry (one image group).

header_features(entry)   46 values (HEADER_FEATURES): physical quantities on comparable scales
                         (roughly 0..1), one missingness flag per quantity (0 is not "missing"),
                         orientation, pixel spacing, coverage, scanning sequence, 2D/3D, ImageType
                         flags, contrast agent.
TextEncoder              SeriesDescription + ScanningSequence + SequenceVariant + BodyPartExamined
                         as one string -> TF-IDF over character n-grams -> TruncatedSVD (LSA). No
                         keywords are hand-picked; the model learns the text -> class mapping. Fit on
                         training rows only and stored with the model.
"""
from __future__ import annotations

import numpy as np

from .dicom_index import META_FIELDS

_F = {name: i for i, name in enumerate(META_FIELDS)}

HEADER_FEATURES = (
    ["log_tr", "log_te", "log_ti", "flip", "field", "log_averages", "slice_thickness", "log_bandwidth",
     "slice_spacing"]
    + [f"missing_{f}" for f in ("tr", "te", "ti", "flip", "field", "averages", "thickness", "bandwidth", "spacing")]
    + [f"orient_{i}" for i in range(6)] + ["missing_orientation"]
    + ["log_n_slices", "pixel_spacing_row", "pixel_spacing_col", "missing_pixel_spacing", "log_coverage"]
    + [f"seq_{s}" for s in ("SE", "GR", "IR", "EP", "RM")]
    + ["acq_3d", "acq_2d"]
    + [f"type_{t}" for t in ("derived", "mpr", "projection", "phase", "adc", "trace", "exp", "localizer")]
    + ["contrast_agent"]
)

TEXT_FIELDS = ("series_description", "scanning_sequence", "sequence_variant", "body_part_examined")


def meta_value(entry: dict, name: str) -> float:
    mv = entry.get("meta_vec")
    if mv is None or len(mv) <= _F[name]:
        return 0.0
    v = float(mv[_F[name]])
    return v if np.isfinite(v) else 0.0


def image_type(entry: dict) -> list[str]:
    it = entry.get("image_type") or []
    it = it.split("\\") if isinstance(it, str) else it
    return [str(x).strip().upper() for x in it]


def header_features(entry: dict | None) -> np.ndarray:
    e = entry or {}
    raw = {n: meta_value(e, n) for n in META_FIELDS}
    log = lambda v, s: np.log1p(max(v, 0.0)) / s
    vals = [log(raw["RepetitionTime"], 10), log(raw["EchoTime"], 6), log(raw["InversionTime"], 8),
            raw["FlipAngle"] / 180, raw["MagneticFieldStrength"] / 3, log(raw["NumberOfAverages"], 3),
            raw["SliceThickness"] / 10, log(raw["PixelBandwidth"], 8), raw["SpacingBetweenSlices"] / 10]
    missing = [float(raw[n] == 0.0) for n in META_FIELDS]
    orient = [float(x) for x in (e.get("orientation") or [0.0] * 6)][:6]
    ps = [float(x) for x in (e.get("pixel_spacing") or [0.0, 0.0])][:2]
    nsl = float(e.get("n_slices_total", 0) or 0)
    seq = str(e.get("scanning_sequence", "")).upper()
    acq = str(e.get("mr_acquisition_type", "")).upper()
    it = image_type(e)
    has = lambda *toks: float(any(t in x for x in it for t in toks))
    vec = (vals + missing
           + orient + [float(not any(orient))]
           + [log(nsl, 6), ps[0] / 2, ps[1] / 2, float(not any(ps)), log(nsl * raw["SliceThickness"], 6)]
           + [float(s in seq) for s in ("SE", "GR", "IR", "EP", "RM")]
           + [float(acq == "3D"), float(acq == "2D")]
           + [float("DERIVED" in it), has("MPR", "REFORMAT"), has("PROJECTION", "MIP", "MNIP"),
              float(bool(set(it) & {"P", "PHASE", "I", "IMAGINARY"})),
              float(any(x == "ADC" or x.startswith("ADC_") for x in it)),
              has("TRACEW"), float(any(x in ("EXP", "EADC") for x in it)), has("LOCALIZER")]
           + [float(bool(str(e.get("contrast_bolus_agent", "")).strip()))])
    return np.asarray(vec, np.float32)


def header_matrix(index, keys: list[str]) -> np.ndarray:
    X = np.vstack([header_features(index.get(k)) for k in keys]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def texts(index, keys: list[str]) -> list[str]:
    out = []
    for k in keys:
        e = index.get(k) or {}
        out.append(" ".join(str(e.get(f, "") or "") for f in TEXT_FIELDS).strip())
    return out


class TextEncoder:
    """LSA of the description text: TF-IDF (char n-grams) -> TruncatedSVD to `dim` values. A
    degenerate corpus leaves it disabled (all-zero output, layout kept)."""

    def __init__(self, dim: int = 48, ngram=(3, 5), analyzer: str = "char_wb", min_df: int = 2, seed: int = 42):
        self.dim, self.ngram, self.analyzer, self.min_df, self.seed = dim, tuple(ngram), analyzer, min_df, seed
        self.vectorizer = self.svd = None

    def fit(self, corpus: list[str]) -> "TextEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        clean = [t for t in corpus if str(t).strip()]
        if len(clean) < max(self.min_df, 3):
            return self
        vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram, min_df=self.min_df)
        try:
            tfidf = vec.fit_transform(clean)
        except ValueError:          # empty vocabulary
            return self
        n_comp = int(min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        if n_comp < 1:
            return self
        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        try:        # OpenBLAS can deadlock in the SVD after a forked process pool
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                svd.fit(tfidf)
        except ImportError:
            svd.fit(tfidf)
        self.vectorizer, self.svd = vec, svd
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        out = np.zeros((len(corpus), self.dim), dtype=np.float32)
        if self.svd is None:
            return out
        z = self.svd.transform(self.vectorizer.transform([str(t) for t in corpus])).astype(np.float32)
        out[:, :z.shape[1]] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return out
