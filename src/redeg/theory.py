"""Closed forms of the theory (paper Sec. 3, Theorem 2, Corollary 1, Theorem 4) and synthetic power-law fields.

* knee law (Theorem 2e): ``rho(eta) ~ 1 - (1 + eta/s0)^(-gamma)`` for gradient spectra ``c lambda^(gamma-1)`` on
  ``[0, Lambda]``; the error is at most ``(1 + eta/s0)^(-gamma) Q / (1 - Q)``, ``Q`` the regularised upper incomplete
  gamma function at ``(gamma, s0 Lambda)``. An isotropic power spectrum ``|omega|^(-alpha)`` gives ``gamma = 2 - alpha/2``.
* half-energy law (Corollary 1): a re-blur by ``eta_1/2 = s0 (2^(1/gamma) - 1)`` removes half of the gradient energy;
  natural images (``alpha ~ 2``) have ``gamma ~ 1`` and ``eta_1/2 ~ s0``.
* sensitivity (Corollary 1): ``|d rho / d log s0| = gamma u (1 + u)^(-gamma-1)``, ``u = eta / s0``, peaks at ``u = 1/gamma``.
* narrowband obstruction (Theorem 2d): ``|d rho / d s0| <= min(lambda_c, eta lambda_c^2 / 4)`` for spectra on ``[0, lambda_c]``.
* two-point lower bound (Theorem 4): with white noise of standard deviation ``sigma``, scene spectra on ``[0, lambda_c]``
  and ``m2 = E ||x - mean(x)||^2``, no estimator built from ``n`` unlabelled images is accurate to better than
  ``|A(s0 + h) - A(s0)| / 4`` with ``h = 2 sigma / (lambda_c sqrt(n m2))``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaincc

from .semigroup import PixelTables, _tab

__all__ = ["knee_law", "knee_law_error_bound", "gamma_from_alpha", "half_energy_point", "log_sensitivity",
           "narrowband_slope_bound", "two_point_resolution", "powerlaw_coeffs"]


def knee_law(eta, s0: float, gamma: float):
    return 1.0 - (1.0 + np.asarray(eta, float) / s0) ** (-gamma)


def knee_law_error_bound(eta, s0: float, gamma: float, Lambda: float):
    Q = gammaincc(gamma, s0 * Lambda)
    return (1.0 + np.asarray(eta, float) / s0) ** (-gamma) * Q / (1.0 - Q)


def gamma_from_alpha(alpha: float) -> float:
    return 2.0 - alpha / 2.0


def half_energy_point(s0: float, gamma: float) -> float:
    return s0 * (2.0 ** (1.0 / gamma) - 1.0)


def log_sensitivity(u, gamma: float):
    u = np.asarray(u, float)
    return gamma * u * (1.0 + u) ** (-gamma - 1.0)


def narrowband_slope_bound(eta: float, lambda_c: float) -> float:
    return float(min(lambda_c, eta * lambda_c ** 2 / 4.0))


def two_point_resolution(sigma: float, lambda_c: float, n: int, m2: float) -> float:
    return float(2.0 * sigma / (lambda_c * np.sqrt(n * m2)))


def powerlaw_coeffs(alpha: float, n: int, rng: np.random.Generator, channels: int = 1, tables: PixelTables | None = None):
    """DCT coefficients of isotropic ``|omega|^(-alpha)`` fields: ``c ~ N(0, 1) lambda^(-alpha/4)``, DC removed (Cell 2.1)."""
    tab = _tab(tables)
    amp = np.zeros_like(tab.lam)
    amp[tab.lam > 0] = tab.lam[tab.lam > 0] ** (-alpha / 4.0)
    return rng.standard_normal((n, channels, tab.h, tab.w)) * amp
