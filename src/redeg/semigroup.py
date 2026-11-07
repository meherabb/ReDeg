"""The exact heat semigroup on a finite tile and the binned gradient spectrum (paper Sec. 3; notebook Cell 2.1).

``T_s = exp(-s L / 2)`` with ``L = D^T D`` the Laplacian of the pixel grid with reflecting (Neumann) boundaries. The
orthonormal 2-D DCT-II diagonalises ``L`` with eigenvalues ``lambda_{k,l} = 4 sin^2(pi k / 2H) + 4 sin^2(pi l / 2W)``, so
``T_s`` multiplies every DCT coefficient by ``exp(-s lambda / 2)``. ``s`` is the variance (px^2, per axis) of the
Gaussian-equivalent point-spread function, and ``T_s T_t = T_{s+t}`` holds to machine precision.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sps

from .constants import ETA_PIX, HIGH_BAND_LAMBDA_MIN, N_EPS, N_LAMBDA_BINS, TILE

__all__ = ["PixelTables", "tables_for", "dct2", "idct2", "heat", "standardize", "spectra", "spectra_from_coeffs",
           "dct_matrix"]


def dct_matrix(n: int) -> np.ndarray:
    """The orthonormal DCT-II matrix ``Phi^T`` of size n (rows are basis vectors), as in the v1.0 reference code."""
    k = np.arange(n)
    D = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    D[0] /= np.sqrt(2.0)
    return D


class PixelTables:
    """Every size-dependent table of the method for H x W images.

    The formulas are those of the notebook (Cells 2.1, 3.1, 3.2, and ``pixel_tables(N)`` of Cell 6.4c, which builds the
    tables for 32-px CIFAR images): the Laplacian eigenvalues, their 3000 log-spaced bins, the bin centres ``lamb`` and
    counts, the white-noise response ``ew`` of the v1.0 joint fit, the noise band ``lambda >= 6`` and the noise energy
    per re-blur level used by the tail fit.
    """

    def __init__(self, h: int, w: int | None = None):
        w = h if w is None else w
        self.h, self.w = int(h), int(w)
        kh, kw = np.arange(self.h), np.arange(self.w)
        lam_h = 4.0 * np.sin(np.pi * kh / (2 * self.h)) ** 2
        lam_w = 4.0 * np.sin(np.pi * kw / (2 * self.w)) ** 2
        self.lam = lam_h[:, None] + lam_w[None, :]
        lf = self.lam.ravel()
        pos = lf > 0
        edges = np.geomspace(lf[pos].min() * 0.999, lf.max() * 1.001, N_LAMBDA_BINS + 1)
        nb = len(edges) - 1
        b = np.clip(np.searchsorted(edges, lf) - 1, 0, nb - 1)
        cnt = np.bincount(b[pos], minlength=nb).astype(float)
        self.lam_flat, self.pos, self.edges, self.nbins, self.bin_index = lf, pos, edges, nb, b
        self.counts = cnt
        self.lamb = np.bincount(b[pos], weights=lf[pos], minlength=nb) / np.maximum(cnt, 1)
        self.binmat_t = sps.csr_matrix((np.ones(int(pos.sum())), (b[pos], np.nonzero(pos)[0])), shape=(nb, self.h * self.w))
        self.ew = (np.exp(-np.outer(ETA_PIX, self.lamb)) @ (cnt * self.lamb)) / np.sum(cnt * self.lamb)
        self.hi = (self.lamb >= HIGH_BAND_LAMBDA_MIN) & (cnt > 0)
        self.noise_energy = np.exp(-np.outer(ETA_PIX, self.lamb)) @ (cnt * self.lamb)
        self._joint = None

    def __repr__(self) -> str:
        return f"PixelTables({self.h}x{self.w}, {self.nbins} lambda bins, {int((self.counts > 0).sum())} non-empty)"

    def attenuation(self, s: float) -> np.ndarray:
        """``exp(-s lambda / 2)``: the DCT-domain multiplier of ``T_s``."""
        return np.exp(-0.5 * s * self.lam)


@lru_cache(maxsize=8)
def tables_for(h: int = TILE, w: int | None = None) -> PixelTables:
    """Cached :class:`PixelTables` for H x W images (224 x 224 by default)."""
    return PixelTables(h, w)


def _tab(tables: PixelTables | None) -> PixelTables:
    return tables_for(TILE) if tables is None else tables


def dct2(x: np.ndarray) -> np.ndarray:
    """``Phi^T x`` over the last two axes (type-II DCT, orthonormal)."""
    return sfft.dctn(x, type=2, norm="ortho", axes=(-2, -1), workers=-1)


def idct2(c: np.ndarray) -> np.ndarray:
    """Inverse of :func:`dct2`."""
    return sfft.idctn(c, type=2, norm="ortho", axes=(-2, -1), workers=-1)


def heat(x: np.ndarray, s: float, tables: PixelTables | None = None) -> np.ndarray:
    """``T_s x = Phi (exp(-s lambda / 2) Phi^T x)`` over the last two axes; exact on the finite tile.

    ``s`` may be negative for a deblurring step in the DCT domain (used only for finite differences in the tests).
    """
    tab = tables if tables is not None else tables_for(*np.shape(x)[-2:])
    return idct2(dct2(x) * np.exp(-0.5 * s * tab.lam))


def standardize(x: np.ndarray, eps: float | None = None) -> np.ndarray:
    """``N x = (x - mean) / sd`` per image and channel over the last two axes (sd with Bessel's correction).

    ``eps=None`` gives the exact map of Lemma 1 (``N o A = N``); ``eps=1e-6`` the pipeline form of v1.0.
    """
    m = x.mean(axis=(-2, -1), keepdims=True)
    sd = x.std(axis=(-2, -1), ddof=1, keepdims=True)
    return (x - m) / sd if eps is None else (x - m) / (sd + eps)


def spectra_from_coeffs(c: np.ndarray, tables: PixelTables | None = None) -> np.ndarray:
    """Binned gradient-power measure ``nu_x``: ``W[i, ch, b] = sum over bin b of lambda c^2`` (positive lambda only).

    ``c`` holds DCT coefficients of shape (n, channels, H, W); the result has shape (n, channels, 3000).
    """
    tab = _tab(tables)
    n = c.shape[0]
    g = (c.reshape(n * c.shape[1], -1) ** 2) * tab.lam_flat
    return np.asarray(tab.binmat_t @ g.T).T.reshape(n, c.shape[1], tab.nbins)


def spectra(tiles: np.ndarray, bs: int = 32, tables: PixelTables | None = None) -> np.ndarray:
    """``W`` of shape (n, channels, 3000), float64, from tiles of shape (n, H, W, channels) in DN (uint8 or float)."""
    tiles = np.asarray(tiles)
    if tiles.ndim != 4:
        raise ValueError(f"expected tiles of shape (n, H, W, channels), got {tiles.shape}")
    tab = tables if tables is not None else tables_for(tiles.shape[1], tiles.shape[2])
    if (tab.h, tab.w) != tuple(tiles.shape[1:3]):
        raise ValueError(f"tables are {tab.h}x{tab.w} but the tiles are {tiles.shape[1]}x{tiles.shape[2]}")
    out = np.empty((len(tiles), tiles.shape[3], tab.nbins))
    for i in range(0, len(tiles), bs):
        x = np.asarray(tiles[i:i + bs], dtype=np.float64).transpose(0, 3, 1, 2)
        out[i:i + len(x)] = spectra_from_coeffs(dct2(x), tab)
    return out


def standardize_pipeline(x: np.ndarray) -> np.ndarray:
    """The v1.0 pipeline standardisation ``(x - mean) / (sd + 1e-6)``."""
    return standardize(x, eps=N_EPS)
