"""Attribution score ``phi``, the switch and the blend (paper Eq. 3; addendum v6; notebook Cells 6.4g and 6.4k).

    phi = (c_src(0) - c_src(s5, sigma5)) / (c_src(0) - c_tgt)

with ``c`` the mean maximum softmax probability after temperature scaling. ``phi`` is the share of the source-target
confidence gap that re-degrading the labelled source closes; it needs no target label. It is undefined when
``|c_src(0) - c_tgt| < 0.005``. ``phi_7`` uses the storage-matched source in the numerator.
"""
from __future__ import annotations

import numpy as np

from .constants import PHI_MIN_GAP, PHI_SWITCH

__all__ = ["attribution_score", "labelled_share", "switch", "blend"]


def attribution_score(conf_src0: float, conf_src_level: float, conf_tgt: float, min_gap: float = PHI_MIN_GAP) -> float:
    """``phi`` from three mean confidences; NaN when the source-target gap is below ``min_gap``."""
    c0, c5, ct = float(conf_src0), float(conf_src_level), float(conf_tgt)
    return (c0 - c5) / (c0 - ct) if abs(c0 - ct) >= min_gap else np.nan


def labelled_share(acc_src0: float, acc_src_level: float, acc_tgt: float, min_drop: float = 0.02) -> float:
    """Evaluation only (uses target labels): ``(A_src(0) - A_src(s5)) / (A_src(0) - A_tgt)`` on cells whose labelled drop
    exceeds 2 pp, else NaN. Values above 1 mean that the part of the shift that is not blur raised accuracy."""
    a0, a5, at = float(acc_src0), float(acc_src_level), float(acc_tgt)
    return (a0 - a5) / (a0 - at) if (a0 - at) > min_drop else np.nan


def switch(x: float, x_plugin: float, phi: float, threshold: float = PHI_SWITCH) -> float:
    """``X_sw``: the plug-in when ``phi >= 0.5`` (blur explains most of the drop), else the published estimator."""
    on = bool(np.isfinite(phi) and phi >= threshold)
    return x_plugin if (on and np.isfinite(x_plugin)) else x


def blend(x: float, x_plugin: float, phi: float) -> float:
    """``X_bl = X + clip(phi, 0, 1) (X@s_hat - X)`` (reported in v6; weight 0 when ``phi`` is undefined)."""
    w = float(np.clip(phi, 0.0, 1.0)) if np.isfinite(phi) else 0.0
    return (x + w * (x_plugin - x)) if np.isfinite(x_plugin) else x
