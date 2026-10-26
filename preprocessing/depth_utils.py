"""preprocessing/depth_utils.py — depth normalisation + cache I/O (shared by preprocessing and training)."""
import os
from pathlib import Path

import cv2
import numpy as np


def depth_path(cache_root, image_path):
    """'VOC2007/JPEGImages/000005.jpg' -> '<cache_root>/VOC2007/000005.png'"""
    p = Path(image_path)
    return Path(cache_root) / p.parts[0] / (p.stem + ".png")


def normalize_depth(raw, lo_pct=2.0, hi_pct=98.0):
    """Per-image robust min-max to [0, 1]. Convention: larger value = nearer (inverse relative depth).
    Non-finite values are replaced by the median. Returns (depth01, raw_lo, raw_hi, n_nonfinite)."""
    raw = np.asarray(raw, dtype=np.float32)
    bad = ~np.isfinite(raw)
    n_bad = int(bad.sum())
    if n_bad:
        good = raw[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        raw = np.where(bad, fill, raw)
    lo, hi = np.percentile(raw, [lo_pct, hi_pct])
    full = float(raw.max() - raw.min())
    if hi - lo < 0.01 * full:  # clipping would erase the only structure (e.g. small object on empty sky)
        lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-6:  # truly flat map
        return np.zeros_like(raw), float(lo), float(hi), n_bad
    d = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    return d, float(lo), float(hi), n_bad


def save_depth(path, d01):
    """Save a [0,1] float map as a 16-bit PNG (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    img = np.clip(np.rint(d01 * 65535.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(tmp), img):
        raise IOError(f"could not write {tmp}")
    os.replace(tmp, path)


def load_depth(path, expected_hw=None):
    """Load a cached depth map as float32 in [0,1]. Returns None if the file is missing.
    Raises ValueError on wrong dtype or if the shape does not match expected_hw=(H, W)."""
    if not Path(path).exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable depth file {path}")
    if img.ndim != 2 or img.dtype != np.uint16:
        raise ValueError(f"{path}: expected 2-D uint16, got {img.dtype} {img.shape}")
    if expected_hw is not None and tuple(img.shape) != tuple(expected_hw):
        raise ValueError(f"{path}: depth shape {img.shape} != image shape {tuple(expected_hw)}")
    return img.astype(np.float32) / 65535.0
