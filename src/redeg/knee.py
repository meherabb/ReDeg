"""Knee fits of the response curve (paper Sec. 4; notebook Cells 3.1 and 3.2).

* :func:`knee_fit` - the frozen v1.0 fit of ``rho(eta) = 1 - (1 + eta/s)^(-gamma)`` by grid least squares over
  ``S_FIT x G_FIT`` on the 13 re-blur levels; returns the knee ``s_raw``, the exponent and the residual sum of squares.
* :func:`knee_fit_noise` - the v1.0 joint blur + white-noise fit, ``rho = 1 - [(1 + eta/s)^(-gamma) + r ew(eta)] / (1 + r)``.
* :func:`tail_fit` - the v1.1 noise-profiled tail fit adopted by addendum v2: the noise variance is read from the band
  ``lambda >= 6``, its energy is subtracted, and ``log E(eta) = c - gamma log(s + eta)`` is fitted on the levels where the
  noise is below 10% of the observed energy (at least four levels).

All three are the vectorised forms that the notebook asserts equal to its v1.0 reference loops; the loops are kept in the
tests (``tests/reference.py``).
"""
from __future__ import annotations

import numpy as np

from .constants import ETA_PIX, G2, G_FIT, R_FIT, S2, S_FIT
from .semigroup import PixelTables, _tab

__all__ = ["knee_fit", "knee_fit_noise", "tail_fit", "KNEE_GRID"]

_BASE = (1 + ETA_PIX[None, None, :] / S_FIT[:, None, None]) ** (-G_FIT[None, :, None])   # (170, 39, 13)
KNEE_GRID = np.c_[np.repeat(S_FIT, len(G_FIT)), np.tile(G_FIT, len(S_FIT))]              # (s, gamma) of each model
_M1 = (1 - _BASE).reshape(-1, len(ETA_PIX)).astype(np.float64)
_X2 = np.log(S2[:, None] + ETA_PIX[None, :])


def knee_fit(R: np.ndarray, cols=None, chunk: int = 1024):
    """Grid least squares of the knee law on response curves ``R`` (n, 13) -> (s_raw, gamma, SSE), each of shape (n,).

    ``cols`` restricts the fit to a subset of the re-blur levels.
    """
    R = np.asarray(R, np.float64)
    M = _M1 if cols is None else _M1[:, cols]
    Rc = R if cols is None else R[:, cols]
    Msq = (M ** 2).sum(1)
    s_, g_, e_ = np.empty(len(Rc)), np.empty(len(Rc)), np.empty(len(Rc))
    for i in range(0, len(Rc), chunk):
        r = Rc[i:i + chunk]
        sse = (r ** 2).sum(1)[:, None] - 2 * r @ M.T + Msq[None, :]
        j = sse.argmin(1)
        s_[i:i + chunk], g_[i:i + chunk] = KNEE_GRID[j, 0], KNEE_GRID[j, 1]
        e_[i:i + chunk] = np.maximum(sse[np.arange(len(r)), j], 0)
    return s_, g_, e_


def _joint_models(tab: PixelTables):
    if tab._joint is None:
        mn = [(1 - (_BASE + r_ * tab.ew[None, None, :]) / (1 + r_)).reshape(-1, len(ETA_PIX)).astype(np.float64) for r_ in R_FIT]
        tab._joint = (mn, [(M ** 2).sum(1) for M in mn])
    return tab._joint


def knee_fit_noise(R: np.ndarray, tables: PixelTables | None = None, chunk: int = 1024):
    """v1.0 joint blur + fresh-white-noise fit -> (s, noise-to-signal ratio r, SSE), each of shape (n,)."""
    MN, MNSQ = _joint_models(_tab(tables))
    R = np.asarray(R, np.float64)
    bs, br, be = np.empty(len(R)), np.zeros(len(R)), np.full(len(R), np.inf)
    for i in range(0, len(R), chunk):
        r = R[i:i + chunk]
        r2 = (r ** 2).sum(1)[:, None]
        for k, (M, Msq) in enumerate(zip(MN, MNSQ)):
            sse = r2 - 2 * r @ M.T + Msq[None, :]
            j = sse.argmin(1)
            m = sse[np.arange(len(r)), j]
            u = m < be[i:i + chunk]
            be[i:i + chunk][u] = m[u]
            bs[i:i + chunk][u] = KNEE_GRID[j[u], 0]
            br[i:i + chunk][u] = R_FIT[k]
    return bs, br, be


def _tail_prep(W, sig2, eta_min, tab: PixelTables):
    Eobs = W.sum(1) @ np.exp(-np.outer(ETA_PIX, tab.lamb)).T
    if sig2 is None:
        sig2 = np.median(W[..., tab.hi] / (tab.counts[tab.hi] * tab.lamb[tab.hi]), axis=-1).mean(1)
    En = W.shape[1] * sig2[:, None] * tab.noise_energy[None, :]
    Es = Eobs - En
    valid = (En <= 0.1 * Eobs) & (Es > 0)
    if eta_min is not None:
        valid &= ETA_PIX[None, :] >= eta_min
    return np.log(np.maximum(Es, 1e-30)), valid, sig2


def tail_fit(W: np.ndarray, sig2=None, eta_min=None, chunk: int = 512, tables: PixelTables | None = None):
    """v1.1 tail fit -> (s per image, smallest valid eta per image, noise variance per image).

    Images with fewer than four valid levels get NaN. With ``sig2`` given, that noise variance is used instead of the one
    read from the band ``lambda >= 6``; ``eta_min`` restricts the fit to levels at or above it (used to calibrate on the
    source over the same range as the target).
    """
    tab = _tab(tables)
    L, valid, sig2 = _tail_prep(W, sig2, eta_min, tab)
    out, emin = np.full(len(W), np.nan), np.full(len(W), np.nan)
    ok = np.nonzero(valid.sum(1) >= 4)[0]
    if len(ok) == 0:
        return out, emin, sig2
    pats, inv_idx = np.unique(valid[ok], axis=0, return_inverse=True)
    inv_idx = np.asarray(inv_idx).ravel()
    for pi, pat in enumerate(pats):
        rows = ok[inv_idx == pi]
        Xc = _X2[:, pat] - _X2[:, pat].mean(1, keepdims=True)
        SXX = (Xc ** 2).sum(1)
        for i in range(0, len(rows), chunk):
            rr = rows[i:i + chunk]
            Lc = L[rr][:, pat] - L[rr][:, pat].mean(1, keepdims=True)
            sse = ((Lc ** 2).sum(1)[:, None, None] + 2 * G2[None, None, :] * (Lc @ Xc.T)[:, :, None]
                   + (G2 ** 2)[None, None, :] * SXX[None, :, None])
            j = sse.reshape(len(rr), -1).argmin(1)
            out[rr], emin[rr] = S2[j // len(G2)], ETA_PIX[pat].min()
    return out, emin, sig2
