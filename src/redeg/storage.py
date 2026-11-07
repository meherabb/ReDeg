"""Storage-matched re-degradation (addendum v7; paper Sec. 4; notebook Cells 5.11 and 6.4k).

A target that was blurred and then stored is not ``T_s x``: rounding to 8 bits and JPEG compression leave a
high-frequency floor that a floating-point re-degradation lacks. The labelled source is therefore re-degraded as

    D(x) = decode(JPEG_{q,c}(Q8(T_s x + sigma xi)))

with the JPEG quality ``q`` and the chroma subsampling ``c`` read from the quantisation tables and sampling factors of the
target files (no label is used). The JPEG step is dropped for lossless 8-bit targets, and ``Q8`` for floating-point ones.
"""
from __future__ import annotations

import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, JpegImagePlugin

from ._keys import rng_for
from .semigroup import PixelTables, heat, tables_for

__all__ = ["pillow_qtables", "jpeg_file_store", "read_store", "store_label", "jpeg_roundtrip", "jpeg_bytes", "quantize8",
           "redegrade", "storage_is_exact"]

_QTAB: dict = {}
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def pillow_qtables() -> dict:
    """Pillow's quantisation tables (luminance, chrominance) for every quality 1-100 (IJG scaling of the standard tables)."""
    if not _QTAB:
        for q in range(1, 101):
            buf = io.BytesIO()
            Image.new("RGB", (16, 16), (128, 96, 160)).save(buf, format="JPEG", quality=q)
            with Image.open(io.BytesIO(buf.getvalue())) as im:
                t = im.quantization
            _QTAB[q] = (tuple(int(v) for v in t[0]), tuple(int(v) for v in t.get(1, t[0])))
    return _QTAB


def jpeg_file_store(src):
    """``(quality, subsampling, exact table match)`` of one image file (path, file object or bytes); None if not a JPEG.

    The quality is the one whose IJG tables equal the file's; without an exact match, the nearest luminance table in L1.
    Subsampling codes: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0.
    """
    with Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src) as im:
        if im.format != "JPEG":
            return None
        qt = im.quantization
        lum, chrom = tuple(int(v) for v in qt[0]), tuple(int(v) for v in qt.get(1, qt[0]))
        sub = int(JpegImagePlugin.get_sampling(im))
    tabs = pillow_qtables()
    exact = [q for q, (a, b) in tabs.items() if a == lum and b == chrom]
    if exact:
        return exact[0], sub, True
    q = min(tabs, key=lambda q: sum(abs(x - y) for x, y in zip(tabs[q][0], lum)))
    return q, sub, False


def read_store(items, kind_hint: str | None = None, cap: int | None = None, key=("store",)) -> dict:
    """The storage step of a target set, read from its files: ``dict(kind, q, sub, exact, n, share)``.

    ``items`` are paths, file objects or JPEG byte strings. The modal (quality, subsampling) pair is returned with the share
    of files that use it. ``kind_hint`` declares arrays: ``"lossless8"`` (uint8 arrays or PNG files) or ``"float"``.
    """
    if kind_hint == "float":
        return dict(kind="float")
    if kind_hint == "lossless8":
        return dict(kind="lossless8", n=int(len(items)) if items is not None else 0)
    sel = list(range(len(items)))
    if cap is not None and len(sel) > cap:
        sel = sorted(rng_for(*key).permutation(len(items))[:cap].tolist())
    rec = [jpeg_file_store(items[i]) for i in sel]
    jp = [r for r in rec if r is not None]
    if not jp:
        return dict(kind="lossless8", n=len(rec))
    pairs = Counter((q, s) for q, s, _ in jp)
    (q, s), cnt = pairs.most_common(1)[0]
    return dict(kind="jpeg", q=int(q), sub=int(s), n=len(rec), share=cnt / len(rec), exact=float(np.mean([e for _, _, e in jp])),
                non_jpeg=len(rec) - len(jp))


def store_label(st: dict | None) -> str:
    if not st:
        return "?"
    if st.get("kind") == "jpeg":
        return f"JPEG q{st['q']} {SUBSAMPLING.get(st.get('sub'), 'sub?')}"
    return st.get("kind", "?")


def jpeg_bytes(img_u8: np.ndarray, q: int = 85, sub: int | None = None) -> bytes:
    """Encode one uint8 image (H, W, 3) as JPEG with Pillow, ``optimize=True``, as the notebook stored its JPEG 85 targets
    (Cell 5.10). ``sub=None`` keeps Pillow's default subsampling (4:2:0)."""
    kw = dict(format="JPEG", quality=int(q), optimize=True)
    if sub is not None:
        kw["subsampling"] = int(sub)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img_u8)).save(buf, **kw)
    return buf.getvalue()


def jpeg_roundtrip(X: np.ndarray, q: int, sub: int, threads: int = 4) -> np.ndarray:
    """``decode(JPEG_{q,sub}(X))`` for a uint8 batch (n, H, W, 3) with Pillow (``optimize=True``), as in Cell 5.11."""
    out = np.empty_like(X)

    def one(i):
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(X[i])).save(buf, format="JPEG", quality=int(q), subsampling=int(sub), optimize=True)
        with Image.open(io.BytesIO(buf.getvalue())) as im:
            out[i] = np.asarray(im.convert("RGB"))

    with ThreadPoolExecutor(max(1, int(threads))) as ex:
        list(ex.map(one, range(len(X))))
    return out


def quantize8(y: np.ndarray) -> np.ndarray:
    """``Q8``: round to the nearest integer and clip to [0, 255] (uint8)."""
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def storage_is_exact(s: float, sigma: float, store: dict | None) -> bool:
    """True when the storage-matched source equals the floating-point one (float targets) or the clean source itself
    (lossless targets at ``s = sigma = 0``); the v7 plug-ins then reduce to the v6 forms."""
    k = (store or {}).get("kind", "float")
    return k == "float" or (k == "lossless8" and float(s) == 0.0 and float(sigma) == 0.0)


def redegrade(tiles: np.ndarray, s: float, sigma: float = 0.0, store: dict | None = None, key=("redegrade",), bs: int = 128,
              tables: PixelTables | None = None, threads: int = 4) -> np.ndarray:
    """Re-degrade source tiles (n, H, W, C) in DN to the identified level and through the target's storage step.

    * ``store=None`` or ``{"kind": "float"}``: ``T_s x + sigma xi`` as float64 (the floating-point re-degradation of v6);
    * ``{"kind": "lossless8"}``: ``Q8(T_s x + sigma xi)`` as uint8;
    * ``{"kind": "jpeg", "q": q, "sub": c}``: ``decode(JPEG_{q,c}(Q8(T_s x + sigma xi)))`` as uint8.

    The noise ``xi`` comes from the keyed stream ``(*key, batch)``. (The notebook draws it with a keyed torch generator on
    the GPU and blurs in float32, so individual noise realisations and a few rounding ties differ from this NumPy version.)
    """
    tiles = np.asarray(tiles)
    tab = tables if tables is not None else tables_for(tiles.shape[1], tiles.shape[2])
    kind = (store or {}).get("kind", "float")
    out = np.empty(tiles.shape, np.float64 if kind == "float" else np.uint8)
    for i in range(0, len(tiles), bs):
        x = np.asarray(tiles[i:i + bs], dtype=np.float64).transpose(0, 3, 1, 2)
        y = heat(x, float(s), tab) if float(s) > 0 else x
        if sigma > 0:
            y = y + float(sigma) * rng_for(*key, i // bs).standard_normal(y.shape)
        y = y.transpose(0, 2, 3, 1)
        out[i:i + len(x)] = y if kind == "float" else quantize8(y)
    if kind == "jpeg":
        out = jpeg_roundtrip(out, store["q"], store["sub"], threads=threads)
    elif kind not in ("float", "lossless8"):
        raise ValueError(f"unknown storage kind {kind!r}")
    return out
