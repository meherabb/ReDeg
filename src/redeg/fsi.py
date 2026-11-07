"""Floor-sensitivity index (paper Eq. 4; addendum v7; notebook Cells 6.4k and 6.4m).

    FSI = 1/3 sum_{s in {1, 4, 16}} 100 [ g(T_s P + 3 DN) - g(T_s P) ],

with ``g`` the confidence-accuracy gap (mean maximum softmax probability minus accuracy, after temperature scaling) of a
classifier on its labelled source blurred to ``s`` by the exact semigroup, with and without 3 DN of white noise. It needs no
target data. An index of at least 4 pp predicts that a floating-point re-degradation will raise the estimators' error
(registered in v7; tested on eight unseen classifiers in v8: 6 of 8 as predicted).
"""
from __future__ import annotations

import numpy as np

from ._keys import rng_for
from .calibration import softmax_T
from .constants import FSI_LEVELS, FSI_THRESHOLD_PP, NOISE_DN
from .semigroup import PixelTables, heat, tables_for

__all__ = ["floor_sensitivity_index", "fsi_inputs", "predicts_harm"]


def floor_sensitivity_index(logits_clean: dict, correct_clean: dict, logits_noisy: dict, correct_noisy: dict,
                            T: float = 1.0, levels=FSI_LEVELS) -> dict:
    """FSI (pp) from the source logits and correctness at each level, without (``*_clean``) and with 3 DN noise
    (``*_noisy``); dicts are keyed by the level ``s`` (float). Returns ``dict(fsi=..., per_level={s: ...})``."""
    per = {}
    for s in levels:
        s = float(s)
        g0 = float(softmax_T(logits_clean[s], T).max(1).mean()) - float(np.mean(correct_clean[s]))
        g3 = float(softmax_T(logits_noisy[s], T).max(1).mean()) - float(np.mean(correct_noisy[s]))
        per[s] = 100.0 * (g3 - g0)
    return dict(fsi=float(np.mean(list(per.values()))), per_level=per)


def fsi_inputs(tiles: np.ndarray, s: float, noise_dn: float = NOISE_DN, key=("fsi",), tables: PixelTables | None = None):
    """The two floating-point inputs of one FSI level: ``T_s x`` and ``T_s x + 3 xi`` (DN scale, shape (n, H, W, C)).

    Feed both to the classifier with its usual input normalisation, without rounding or clipping, as the notebook does."""
    x = np.asarray(tiles, np.float64).transpose(0, 3, 1, 2)
    tab = tables if tables is not None else tables_for(x.shape[2], x.shape[3])
    y = heat(x, float(s), tab) if float(s) > 0 else x
    yn = y + float(noise_dn) * rng_for(*key, float(s)).standard_normal(y.shape)
    return y.transpose(0, 2, 3, 1), yn.transpose(0, 2, 3, 1)


def predicts_harm(fsi: float, threshold: float = FSI_THRESHOLD_PP) -> bool:
    """The registered prediction: a floating-point re-degradation will raise the estimators' error when FSI >= 4 pp."""
    return bool(fsi >= threshold)
