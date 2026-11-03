"""Image input of the image model: N evenly spaced slices of an image group, rescaled, unified to
one matrix, z-normalised, resampled to H x W, cached as .npy (cache key: md5 of key, N, size)."""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

Z_NORM_MIN_STD = 1e-3       # volumes with a smaller std are blank / degenerate


def _apply_rescale(img: np.ndarray, ds) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img * slope + intercept if slope != 1.0 or intercept != 0.0 else img


def _unify_slices(slices: list[np.ndarray]) -> list[np.ndarray]:
    """Crop / pad all 2-D slices to the most common (H, W) (mixed-matrix series)."""
    shapes = [s.shape for s in slices]
    th, tw = max(set(shapes), key=shapes.count)
    out = []
    for s in slices:
        h, w = s.shape
        if h > th:
            s = s[(h - th) // 2:(h - th) // 2 + th, :]
        if w > tw:
            s = s[:, (w - tw) // 2:(w - tw) // 2 + tw]
        ph, pw = th - s.shape[0], tw - s.shape[1]
        if ph > 0 or pw > 0:
            s = np.pad(s, ((0, ph), (0, pw)))
        out.append(s)
    return out


def _frames(img: np.ndarray) -> list[np.ndarray]:
    """2-D frames of one DICOM's pixel array (single frame, multi-frame, RGB)."""
    if img.ndim == 4:                                   # (N, H, W, C) RGB multi-frame
        img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
               if img.shape[-1] >= 3 else img[..., 0])
        return list(img)
    if img.ndim == 3:
        return [img[:, :, 0]] if img.shape[2] <= 4 else list(img)   # (H, W, C) RGB | (N, H, W)
    return [img] if img.ndim == 2 else []


def load_volume(key: str, index, n_slices: int = 16) -> Optional[np.ndarray]:
    """(n_slices, H, W) float32 of an index key, slices sampled evenly by InstanceNumber; None on failure."""
    entry = index.get(key)
    if entry is None or not entry["files"]:
        return None
    files = entry["files"]
    d = len(files)
    chosen = np.linspace(0, d - 1, n_slices, dtype=int).tolist() if d >= n_slices else [i % d for i in range(n_slices)]
    slices: list[np.ndarray] = []
    try:
        with index.source.reader(entry["study"]) as r:
            for i in chosen:
                try:
                    with r.open(files[i][1]) as f:
                        ds = pydicom.dcmread(f)
                    if "PixelData" not in ds:
                        continue
                    img = ds.pixel_array.astype(np.float32)
                except Exception:
                    continue
                slices.extend(_apply_rescale(fr, ds) for fr in _frames(img))
    except Exception:
        return None
    if not slices:
        return None
    slices = _unify_slices(slices)
    while len(slices) < n_slices:
        slices.append(slices[len(slices) % len(slices)])
    return np.stack(slices[:n_slices])


def z_normalise(vol: np.ndarray) -> Optional[np.ndarray]:
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    std = vol.std()
    if std < Z_NORM_MIN_STD:
        return None
    vol = (vol - vol.mean()) / std
    return vol.astype(np.float32) if np.isfinite(vol).all() else None


def resample(vol: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(vol).unsqueeze(1)                 # (N, 1, h, w)
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False).squeeze(1).numpy()


def augment(vol: np.ndarray) -> np.ndarray:
    """Flips, gamma and noise (training only)."""
    if np.random.rand() > 0.5:
        vol = vol[:, :, ::-1].copy()
    if np.random.rand() > 0.5:
        vol = vol[:, ::-1, :].copy()
    g = np.random.uniform(0.75, 1.33)
    mn, rng = vol.min(), vol.max() - vol.min() + 1e-8
    vol = ((vol - mn) / rng) ** g * rng + mn
    sigma = np.random.uniform(0, 0.02) * (vol.std() + 1e-8)
    return (vol + np.random.normal(0, sigma, vol.shape)).astype(np.float32)


class VolumeLoader:
    """Model input of an index key, from the .npy cache or the DICOM source."""

    def __init__(self, index, n_slices: int = 16, size: int = 224, cache_dir: Optional[str] = None):
        self.index, self.n_slices, self.size, self.cache_dir = index, n_slices, size, cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.md5(f"{key}_{self.n_slices}_{self.size}".encode()).hexdigest()
        return os.path.join(self.cache_dir, h + ".npy")

    def __call__(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if path and os.path.exists(path):
            try:
                return np.load(path)
            except (EOFError, ValueError, OSError):      # truncated write: regenerate
                os.remove(path)
        raw = load_volume(key, self.index, self.n_slices)
        vol = z_normalise(raw) if raw is not None else None
        if vol is None:
            return None
        vol = resample(vol, self.size)
        if path:
            np.save(path, vol)
        return vol
