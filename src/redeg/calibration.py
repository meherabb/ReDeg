"""Softmax and temperature scaling (addendum v6; notebook Cells 6.4c and 6.4g).

One temperature per classifier is fitted by minimising the mean negative log-likelihood of the clean labelled source
(Guo et al., ICML 2017), with a bounded scalar search over ``log T`` in ``[log 0.05, log 20]`` (tolerance 1e-4). Every
softmax of every estimator, plug-in and score then uses ``Z / T``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import TEMPERATURE_RANGE

__all__ = ["softmax", "softmax_T", "fit_temperature"]


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax of logits (n, K)."""
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def softmax_T(Z: np.ndarray, T: float = 1.0) -> np.ndarray:
    """``softmax(Z / T)`` in float64."""
    return softmax(np.asarray(Z, np.float64) / float(T))


def fit_temperature(Z: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None):
    """Temperature minimising the mean NLL of labelled logits -> ``(T, NLL at T = 1, NLL at T)``.

    ``y`` holds class indices (columns of ``Z``); ``valid`` masks rows whose label is one of the classifier's classes.
    With fewer than 10 valid rows the temperature is 1.
    """
    valid = np.ones(len(y), bool) if valid is None else np.asarray(valid, bool)
    Z = np.asarray(Z, np.float64)[valid]
    y = np.asarray(y)[valid]
    if len(y) < 10:
        return 1.0, np.nan, np.nan

    def nll(lt):
        S = Z / math.exp(lt)
        m = S.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(S - m).sum(1))
        return float(np.mean(lse - S[np.arange(len(y)), y]))

    lo, hi = TEMPERATURE_RANGE
    r = minimize_scalar(nll, bounds=(math.log(lo), math.log(hi)), method="bounded", options=dict(xatol=1e-4))
    return float(math.exp(r.x)), nll(0.0), float(r.fun)
