"""Published label-free accuracy estimators, as implemented for the paper (notebook Cells 3.3, 4.1, 4.2 and 6.4g).

All functions take softmax probabilities ``P`` (n, K) (after temperature scaling, see :mod:`redeg.calibration`) or logits
``Z``, and return an accuracy estimate in [0, 1]. "Source" is the labelled calibration split; nothing here reads a target
label. COT and COTT need the POT package (``pip install pot``).

* AC (Hendrycks and Gimpel, ICLR 2017): mean maximum softmax probability.
* DoC (Guillory et al., ICCV 2021): source accuracy minus the drop of the mean confidence.
* ATC and ATC-NE (Garg et al., ICLR 2022): share of target scores above the threshold at which the source score
  distribution has the source error rate; scores are the maximum probability or the negative entropy.
* NucNorm (Deng et al., ICML 2023): normalised nuclear norm of the prediction matrix (the score; the accuracy comes from a
  linear regression on meta-sets, Cell 4.1).
* COT and COTT (Lu et al., NeurIPS 2023): exact optimal transport between the predictions and one-hot labels drawn from the
  source label marginal; COT predicts the error as the mean cost (cost 1/2 ||e_y - p||_1), COTT as the share of samples
  whose cost (||e_y - p||_1) exceeds a threshold fitted on the source with its true labels.
* MaNo (Xie et al., NeurIPS 2024): the score ``(1/(N K) sum Q^4)^(1/4)`` of normalised logits.
"""
from __future__ import annotations

import numpy as np

from .constants import MANO_P

__all__ = ["maxconf", "negent", "entropy", "clip01", "atc_predict", "ac", "doc", "atc", "atc_ne", "nuc_score",
           "sample_label_counts", "ot_costs", "cot_cost", "cot", "cott_threshold", "cott", "mano_phi", "mano_score"]


def maxconf(P: np.ndarray) -> np.ndarray:
    return P.max(1)


def negent(P: np.ndarray) -> np.ndarray:
    """Negative entropy ``sum_k p_k log p_k`` (higher = more confident), the ATC-NE score."""
    return np.sum(P * np.log(np.clip(P, 1e-12, None)), axis=1)


def entropy(P: np.ndarray) -> np.ndarray:
    return -negent(P)


def clip01(v) -> float:
    return float(np.clip(v, 0.0, 1.0))


def atc_predict(score_t: np.ndarray, score_src: np.ndarray, acc_src: float) -> float:
    """ATC: threshold ``t = quantile(score_src, 1 - acc_src)`` (linear interpolation); predicted accuracy ``mean(score_t > t)``."""
    return float(np.mean(score_t > np.quantile(score_src, 1.0 - acc_src)))


def ac(P_t: np.ndarray) -> float:
    return float(maxconf(P_t).mean())


def doc(P_t: np.ndarray, P_src: np.ndarray, acc_src: float) -> float:
    return clip01(acc_src - (maxconf(P_src).mean() - maxconf(P_t).mean()))


def atc(P_t: np.ndarray, P_src: np.ndarray, acc_src: float) -> float:
    return atc_predict(maxconf(P_t), maxconf(P_src), acc_src)


def atc_ne(P_t: np.ndarray, P_src: np.ndarray, acc_src: float) -> float:
    return atc_predict(negent(P_t), negent(P_src), acc_src)


def nuc_score(P: np.ndarray) -> float:
    """Normalised nuclear norm ``||P||_* / sqrt(min(n, K) n)`` of the softmax matrix."""
    n, K = P.shape
    return float(np.linalg.norm(P, ord="nuc") / np.sqrt(min(n, K) * n))


def sample_label_counts(prior: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Proportional allocation of ``n`` labels to the classes, plus a random remainder (official ``sample_label_dist``)."""
    base = np.floor(prior * n).astype(int)
    rem = int(n - base.sum())
    if rem > 0:
        base += np.bincount(rng.choice(len(prior), size=rem, p=prior), minlength=len(prior))
    return base


def ot_costs(P: np.ndarray, counts: np.ndarray, full_l1: bool) -> np.ndarray:
    """Exact OT between n one-hot labels (given as class counts) and the n softmax rows -> per-sample transport cost."""
    import ot  # POT, imported lazily: only COT and COTT need it

    n = P.shape[0]
    keep = counts > 0
    a = counts[keep].astype(np.float64)
    a = a / a.sum()
    b = np.full(n, 1.0 / n)
    M = (1.0 - P.T[keep]).astype(np.float64)                 # 1/2 ||e_c - p||_1 = 1 - p_c
    if full_l1:
        M = 2.0 * M                                          # ||e_c - p||_1 = 2 (1 - p_c)
    G = ot.emd(a, b, np.ascontiguousarray(M), numItermax=10_000_000)
    return n * (G * M).sum(0)


def cot_cost(P: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> float:
    """Mean COT transport cost (the COT error estimate before clipping)."""
    return float(ot_costs(P, sample_label_counts(prior, P.shape[0], rng), False).mean())


def cot(P_t: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> float:
    return clip01(1.0 - cot_cost(P_t, prior, rng))


def cott_threshold(P_src: np.ndarray, y_src: np.ndarray) -> float:
    """COTT threshold: per-sample costs ``||e_y - p||_1`` on the source with its true labels; ``quantile(costs, 1 - err)``."""
    costs = ot_costs(P_src, np.bincount(y_src, minlength=P_src.shape[1]), True)
    err = float(np.mean(P_src.argmax(1) != y_src))
    return float(np.quantile(costs, 1.0 - err))


def cott(P_t: np.ndarray, tau: float, prior: np.ndarray, rng: np.random.Generator) -> float:
    costs = ot_costs(P_t, sample_label_counts(prior, P_t.shape[0], rng), True)
    return clip01(1.0 - float(np.mean(costs > tau)))


def mano_phi(Z: np.ndarray) -> float:
    """MaNo's normalisation criterion: the uniform cross-entropy, mean over images and classes of ``-log softmax(Z)``."""
    Z = np.asarray(Z, np.float64)
    m = Z.max(1, keepdims=True)
    lse = m + np.log(np.exp(Z - m).sum(1, keepdims=True))
    return float(np.mean(lse - Z))


def mano_score(Z: np.ndarray, use_softmax: bool) -> float:
    """MaNo score with ``Q = softmax(Z)``, or the second-order Taylor normalisation (``1 + Z + Z^2/2`` minus its row
    minimum, rows summing to 1). The choice is made once per classifier: softmax when ``mano_phi`` of its source exceeds 5."""
    Z = np.asarray(Z, np.float64)
    if use_softmax:
        m = Z.max(1, keepdims=True)
        Q = np.exp(Z - m)
        Q /= Q.sum(1, keepdims=True)
    else:
        V = 1.0 + Z + 0.5 * Z ** 2
        V = V - V.min(1, keepdims=True)
        Q = V / np.maximum(V.sum(1, keepdims=True), 1e-12)
    return float(np.mean(Q ** MANO_P) ** (1.0 / MANO_P))
