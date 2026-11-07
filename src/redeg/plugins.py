"""Plug-in calibration ``X@s_hat`` (paper Sec. 4; addendum v5 forms, notebook Cells 4.4 and 6.4k).

An estimator X that calibrates on labelled source data is calibrated instead on the source re-degraded to the identified
level ``(s5, sigma5)`` (and, from v7 on, through the target's storage step, see :func:`redeg.storage.redegrade`):

* ATC and ATC-NE re-fit their threshold, DoC its offset and COTT its cost threshold on the re-degraded source;
* COT, AE-LS and MaNo are corrected by bias transfer, ``X - [b(s5, sigma5) - b(0, 0)]``, with ``b`` the estimator's signed
  error on the re-degraded source (for COT: transport cost minus the classifier's error there).

Every form returns X exactly when the re-degraded source is the clean source (``s5 = sigma5 = 0`` for floating-point and
lossless targets), provided the same random keys are used; :func:`plugins_at_level` takes care of that.
"""
from __future__ import annotations

import numpy as np

from ._keys import rng_for
from .estimators import atc_predict, clip01, cot_cost, cott_threshold, maxconf, negent, ot_costs, sample_label_counts

__all__ = ["TargetCosts", "plugins_at_level", "bias_transfer"]


class TargetCosts:
    """The target's COT and COTT transport costs, computed once per target set and shared by an estimator and its plug-in
    (the notebook's keyed streams ``(*target_key, "cot")`` and ``(*target_key, "cott")``)."""

    def __init__(self, P_t: np.ndarray, prior: np.ndarray, target_key=("target",)):
        self.cost = cot_cost(P_t, prior, rng_for(*target_key, "cot"))
        self.cott_costs = ot_costs(P_t, sample_label_counts(prior, P_t.shape[0], rng_for(*target_key, "cott")), True)


def _source_ot(P: np.ndarray, y: np.ndarray, prior: np.ndarray, key):
    return dict(cost=cot_cost(P, prior, rng_for("cot_src", *key)), tau=cott_threshold(P, y))


def plugins_at_level(P_t: np.ndarray, P_lvl: np.ndarray, y_src: np.ndarray, P_zero: np.ndarray, *, prior: np.ndarray | None = None,
                     target_costs: TargetCosts | None = None, target_key=("target",), level_key=("lvl", "s5"),
                     zero_key=("lvl", 0.0, 0.0), with_ot: bool = True) -> dict:
    """Published estimators X (calibrated on the clean source ``P_zero``) and their plug-ins X@s_hat (calibrated on the
    re-degraded source ``P_lvl``), for one target set ``P_t``. ``y_src`` are the source labels (class indices).

    Returns a dict with ``ATC, ATC_NE, DoC, COT, COTT`` and ``ATC@, ATC_NE@, DoC@, COT@, COTT@`` (COT and COTT only when
    ``with_ot``; they need POT). ``level_key`` names the random stream of the source transport at the level; pass
    ``zero_key`` when the level is (0, 0) so that the plug-in is the estimator itself.
    """
    y_src = np.asarray(y_src)
    K = P_t.shape[1]
    prior = np.bincount(y_src, minlength=K) / len(y_src) if prior is None else prior
    c0, cl = P_zero.argmax(1) == y_src, P_lvl.argmax(1) == y_src
    acc0, accl = float(c0.mean()), float(cl.mean())
    mx_t, mx0, mxl = maxconf(P_t), maxconf(P_zero), maxconf(P_lvl)
    ne_t = negent(P_t)
    out = {
        "ATC": atc_predict(mx_t, mx0, acc0), "ATC@": atc_predict(mx_t, mxl, accl),
        "ATC_NE": atc_predict(ne_t, negent(P_zero), acc0), "ATC_NE@": atc_predict(ne_t, negent(P_lvl), accl),
        "DoC": clip01(acc0 - (mx0.mean() - mx_t.mean())), "DoC@": clip01(accl - (mxl.mean() - mx_t.mean())),
        "acc_src0": acc0, "acc_src_level": accl,
    }
    if with_ot:
        tc = target_costs if target_costs is not None else TargetCosts(P_t, prior, target_key)
        q0 = _source_ot(P_zero, y_src, prior, zero_key)
        ql = q0 if tuple(level_key) == tuple(zero_key) else _source_ot(P_lvl, y_src, prior, level_key)
        b0, bl = q0["cost"] - (1.0 - acc0), ql["cost"] - (1.0 - accl)
        out.update({
            "COT": clip01(1.0 - tc.cost), "COT@": clip01((1.0 - tc.cost) + (bl - b0)),
            "COTT": clip01(1.0 - float(np.mean(tc.cott_costs > q0["tau"]))),
            "COTT@": clip01(1.0 - float(np.mean(tc.cott_costs > ql["tau"]))),
        })
    return out


def bias_transfer(x_raw: float, bias_level: float, bias_zero: float) -> float:
    """``clip01(X - [b(s5, sigma5) - b(0, 0)])``: the v6 adapter of AE-LS and MaNo (``x_raw`` is the unclipped estimate)."""
    return clip01(x_raw - (bias_level - bias_zero))
