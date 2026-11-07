"""The re-degradation response and the spectral quantities of Theorem 2 (paper Sec. 3; notebook Cells 2.1 and 2.3).

With ``W`` the binned gradient-power measure of an image (see :func:`redeg.semigroup.spectra`), the gradient energy left
after a further blur ``t`` is ``E(t) = sum_b W_b exp(-t lambda_b)`` and the response is

    rho(eta) = 1 - E(s + eta) / E(s),

computed per channel and averaged over channels. ``rho`` is invariant to per-image gain and offset (Lemma 1).
"""
from __future__ import annotations

import numpy as np

from .constants import ETA_PIX
from .semigroup import PixelTables, _tab

__all__ = ["rho_curves", "drho_ds0", "drho_ds0_fd", "observability", "narrowband_index", "half_point"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def rho_curves(W: np.ndarray, s: float = 0.0, etas=None, tables: PixelTables | None = None) -> np.ndarray:
    """Response curves ``rho(eta)`` of shape (n, K) for images with spectra ``W`` (n, channels, 3000), blurred by ``s`` first.

    ``tables`` must be the tables of the image size the spectra were computed at (224 x 224 by default).
    """
    lamb = _tab(tables).lamb
    etas = ETA_PIX if etas is None else np.asarray(etas, float)
    E0 = W @ np.exp(-s * lamb)
    Ek = W @ np.exp(-np.outer(s + etas, lamb)).T
    return (1.0 - Ek / E0[..., None]).mean(1)


def drho_ds0(W: np.ndarray, etas, tables: PixelTables | None = None) -> np.ndarray:
    """Theorem 2(b): ``d rho / d s0 = -(1 - rho(eta)) int_0^eta Var_{pi_u}(lambda) du``, from the observed spectrum alone.

    Uses the exact log-partition form ``int_0^eta Var_{pi_u} du = E_{pi_0}[lambda] - E_{pi_eta}[lambda]``.
    """
    lamb = _tab(tables).lamb
    etas = np.asarray(etas, float)
    Z0, M0 = W.sum(-1), W @ lamb
    ex = np.exp(-np.outer(etas, lamb))
    Ek, Mk = W @ ex.T, W @ (ex * lamb).T
    return (-(Ek / Z0[..., None]) * ((M0 / Z0)[..., None] - Mk / Ek)).mean(1)


def drho_ds0_fd(W: np.ndarray, etas, h: float = 1e-3, tables: PixelTables | None = None) -> np.ndarray:
    """Central finite difference of ``rho`` in the existing blur (exact operators, so ``T_{-h}`` is defined)."""
    return (rho_curves(W, h, etas, tables) - rho_curves(W, -h, etas, tables)) / (2 * h)


def observability(W: np.ndarray, eta_star: float = 2.0, n_u: int = 32, tables: PixelTables | None = None) -> np.ndarray:
    """``O_x = int_0^{eta*} Var_{pi_u}(lambda) du`` (trapezoid rule on 32 points, channel mean; frozen v1.0)."""
    lamb = _tab(tables).lamb
    us = np.linspace(0, eta_star, n_u)
    V = []
    for u in us:
        Wu = W * np.exp(-u * lamb)
        Z = Wu.sum(-1)
        m1, m2 = (Wu @ lamb) / Z, (Wu @ lamb ** 2) / Z
        V.append((m2 - m1 ** 2).mean(1))
    return _trapz(np.array(V), us, axis=0)


def narrowband_index(W: np.ndarray) -> np.ndarray:
    """``NB_x = 1 - H(p_x) / log 60`` over 60 groups of the 3000 log-lambda bins, channel mean (1 = a single band)."""
    P = W.reshape(W.shape[0], W.shape[1], 60, W.shape[2] // 60).sum(-1)
    p = P / np.maximum(P.sum(-1, keepdims=True), 1e-300)
    H = -(p * np.log(np.where(p > 0, p, 1.0))).sum(-1)
    return (1.0 - H / np.log(60)).mean(1)


def half_point(rho: np.ndarray, etas: np.ndarray) -> float:
    """First ``eta`` at which a response curve reaches 1/2 (log-linear interpolation); NaN if it never does."""
    k = int(np.argmax(rho >= 0.5))
    if rho[k] < 0.5:
        return np.nan
    if k == 0:
        return float(etas[0])
    t = (0.5 - rho[k - 1]) / (rho[k] - rho[k - 1])
    return float(np.exp(np.log(etas[k - 1]) + t * (np.log(etas[k]) - np.log(etas[k - 1]))))
