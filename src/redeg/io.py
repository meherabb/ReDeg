"""Image loading as in the experiments: shortest side to 256 px, centre crop 224, uint8 RGB.

The ten scene datasets were resized with bicubic interpolation (notebook Cell 5.3); the ImageNet blocks used bilinear
interpolation, the ImageNet-C preprocessing (addendum v6). ``resample`` selects between the two.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .constants import TILE

__all__ = ["crop224", "load_tiles", "list_images"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


_RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


def crop224(im: Image.Image, resample: str = "bicubic") -> np.ndarray:
    """One PIL image -> uint8 (224, 224, 3): shortest side resized to 256 (bicubic or bilinear), then centre crop."""
    im = im.convert("RGB")
    w, h = im.size
    sc = 256.0 / min(w, h)
    im = im.resize((max(256, round(w * sc)), max(256, round(h * sc))), _RESAMPLE[resample])
    w, h = im.size
    left, top = (w - TILE) // 2, (h - TILE) // 2
    return np.asarray(im.crop((left, top, left + TILE, top + TILE)), dtype=np.uint8)


def list_images(folder: str | Path) -> list[Path]:
    """Image files of a folder (recursive), sorted."""
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_tiles(paths, drop_degenerate: bool = True, resample: str = "bicubic") -> np.ndarray:
    """uint8 tiles (n, 224, 224, 3) of the given image files. Images with a (near-)constant channel (std < 0.5 DN) are
    dropped, as in the experiments."""
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(crop224(im, resample))
    X = np.stack(out) if out else np.empty((0, TILE, TILE, 3), np.uint8)
    if drop_degenerate and len(X):
        X = X[X.reshape(len(X), -1, 3).std(1).min(1) >= 0.5]
    return X
