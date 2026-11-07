"""Label-free identification of the blur level ``s_hat`` in px^2 (paper Sec. 4; notebook Cells 3.2 and 4.4).

Steps (frozen v1.0, then addendum v2):

1. every target image gets a response curve on the 13 re-blur levels and a knee ``s_raw`` from :func:`redeg.knee.knee_fit`;
2. the labelled source, blurred to every level of ``CAL_GRID``, gives the calibration curve ``CAL(s)`` = median knee;
3. ``s_hat = CAL^{-1}(median target knee)``, inverted on the running maximum of ``CAL``;
4. fit check: a one-sided Mann-Whitney test compares the target's knee-fit residuals with those of the source at
   ``s_hat``; the condition is *flagged* when the blur model fits worse (p < 0.01);
5. v2, flagged conditions only: the tail fit gives ``(s, sigma)``, and a second Mann-Whitney test against the source
   simulated at ``(s, sigma)`` with fresh noise decides whether the noise model is accepted;
6. plug-in level (v5): ``s5`` = the tail-fit level when flagged, else ``s_hat``; ``sigma5`` = the tail-fit noise only when
   the condition is flagged and the noise model is accepted, else 0.

Only pixels enter, so ``s_hat`` is the same for every classifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import mannwhitneyu

from ._keys import rng_for
from .constants import CAL_GRID, GOF_ALPHA, NOISE_MODEL_MAX_REF
from .knee import knee_fit, knee_fit_noise, tail_fit
from .response import rho_curves
from .semigroup import PixelTables, _tab, dct2, spectra, spectra_from_coeffs, tables_for

__all__ = ["Calibration", "calibrate", "inv", "identify_spectra", "identify_tail", "noise_model_pvalue", "plugin_level",
           "Identification", "BlurIdentifier"]


def inv(cal_curve: np.ndarray, v: float) -> float:
    """``CAL^{-1}(v) = interp(v, cummax(CAL), CAL_GRID)`` (frozen v1.0)."""
    return float(np.interp(v, np.maximum.accumulate(cal_curve), CAL_GRID))


def noise_sigma(W: np.ndarray, r: np.ndarray, tables: PixelTables | None = None) -> float:
    """Noise standard deviation (DN) implied by the v1.0 joint fit: ``E r / (1 + r) = sigma^2 sum(cnt lambda)``."""
    tab = _tab(tables)
    E = W.sum(-1).mean(1)
    return float(np.sqrt(np.median(E * r / (1 + r)) / np.sum(tab.counts * tab.lamb)))


@dataclass
class Calibration:
    """Calibration of the labelled source over ``CAL_GRID``: ``cal1`` (median knee) and ``cal2`` (median joint-fit knee)."""
    cal1: np.ndarray
    cal2: np.ndarray | None = None


def calibrate(W_src: np.ndarray, tables: PixelTables | None = None, joint: bool = True) -> Calibration:
    """Calibration curves of the labelled source (spectra ``W_src``), as ``build_cal`` of Cell 3.2."""
    tab = _tab(tables)
    G = len(CAL_GRID)
    cal1, cal2 = np.empty(G), (np.empty(G) if joint else None)
    for i, s in enumerate(CAL_GRID):
        R = rho_curves(W_src, s, tables=tab)
        cal1[i] = np.median(knee_fit(R)[0])
        if joint:
            cal2[i] = np.median(knee_fit_noise(R, tables=tab)[0])
    return Calibration(cal1=cal1, cal2=cal2)


def identify_tail(W_t: np.ndarray, W_src: np.ndarray, tables: PixelTables | None = None):
    """v1.1 tail fit on the target, calibrated on the clean source restricted to the same eta range -> (s, sigma, eta_min)."""
    tab = _tab(tables)
    raw, emin, sig2 = tail_fit(W_t, tables=tab)
    if np.all(np.isnan(raw)):
        return np.nan, float(np.sqrt(np.median(sig2))), np.nan
    em = float(np.nanmedian(emin))
    zero = np.zeros(len(W_src))
    cal = np.array([np.nanmedian(tail_fit(W_src * np.exp(-s * tab.lamb)[None, None, :], sig2=zero, eta_min=em, tables=tab)[0])
                    for s in CAL_GRID])
    cal = np.where(np.isfinite(cal), cal, 0.0)
    return inv(cal, np.nanmedian(raw)), float(np.sqrt(np.median(sig2))), em


def identify_spectra(W: np.ndarray, W_src: np.ndarray, cal: Calibration, tables: PixelTables | None = None,
                     tail: bool = True):
    """Frozen v1.0 identification (steps 1-4) plus, when flagged, the v1.0 joint fit and the v2 tail fit.

    Returns ``(o, knees, residuals)`` with ``o`` a dict in the notebook's names (``s_hat``, ``p_gof``, ``flagged``,
    ``s_use``/``sig_use`` of the v1.0 joint fit, ``s_v11``/``sig_v11`` of the tail fit, ``gamma_med``, ``knee_med``, ``n``).
    """
    tab = _tab(tables)
    R = rho_curves(W, 0.0, tables=tab)
    if W.ndim != 3 or R.shape != (len(W), 13):
        raise ValueError("spectra must have shape (n, channels, 3000)")
    ks, gs, res = knee_fit(R)
    s1 = inv(cal.cal1, np.median(ks))
    p = float(mannwhitneyu(res, knee_fit(rho_curves(W_src, s1, tables=tab))[2], alternative="greater").pvalue)
    o = dict(s_hat=s1, p_gof=p, flagged=bool(p < GOF_ALPHA), s_use=s1, sig_use=0.0, s_v11=s1, sig_v11=0.0,
             gamma_med=float(np.median(gs)), knee_med=float(np.median(ks)), n=int(len(W)))
    if o["flagged"]:
        if cal.cal2 is not None:
            kn, rr, _ = knee_fit_noise(R, tables=tab)
            o["s_use"], o["sig_use"] = inv(cal.cal2, np.median(kn)), noise_sigma(W, rr, tab)
        else:
            o["s_use"], o["sig_use"] = np.nan, np.nan
        if tail:
            o["s_v11"], o["sig_v11"], _ = identify_tail(W, W_src, tab)
    return o, ks, res


def noise_model_pvalue(src_tiles: np.ndarray, res_target: np.ndarray, s: float, sigma: float, key=("target",),
                       max_ref: int = NOISE_MODEL_MAX_REF, tables: PixelTables | None = None) -> float:
    """p-value of the v2 noise-model check: target knee-fit residuals against those of (a keyed subsample of) the
    labelled source simulated at ``(s, sigma)`` with fresh Gaussian noise (one-sided Mann-Whitney, 'greater').

    The noise is added in the DCT domain, which is exact because the orthonormal DCT of white noise is white.
    """
    tab = tables if tables is not None else tables_for(src_tiles.shape[1], src_tiles.shape[2])
    idx = np.sort(rng_for("gof2_subset", *key).permutation(len(src_tiles))[:min(max_ref, len(src_tiles))])
    ref = []
    for b in range(0, len(idx), 32):
        C = dct2(src_tiles[idx[b:b + 32]].astype(np.float64).transpose(0, 3, 1, 2)) * np.exp(-0.5 * s * tab.lam)
        C = C + sigma * rng_for("gof2_noise", *key, b).standard_normal(C.shape)
        ref.append(knee_fit(rho_curves(spectra_from_coeffs(C, tab), 0.0, tables=tab))[2])
    return float(mannwhitneyu(res_target, np.concatenate(ref), alternative="greater").pvalue)


def plugin_level(o: dict) -> tuple[float, float]:
    """``(s5, sigma5)`` of addendum v5 from an identification dict that carries ``flagged2`` (``v5_level`` of Cell 6.4k)."""
    fl = bool(o.get("flagged"))
    s5 = float(o["s_v11"]) if fl else float(o["s_use"])
    g5 = float(o["sig_v11"]) if (fl and not bool(o.get("flagged2", True))) else 0.0
    return s5, g5


@dataclass
class Identification:
    """Result of :meth:`BlurIdentifier.identify` (levels in px^2, noise in DN)."""
    s_hat: float            #: v1.0 level
    p_fit: float            #: p-value of the v1.0 fit check
    flagged: bool           #: the blur-only model fits worse than on the source (p < 0.01)
    s_tail: float           #: v2 tail-fit level (= s_hat when not flagged; NaN when no image has four valid levels)
    sigma_tail: float       #: v2 tail-fit noise (DN; 0 when not flagged)
    p_noise: float          #: p-value of the v2 noise-model check (= p_fit when not flagged; NaN when not run)
    noise_rejected: bool    #: the v2 check rejects the blur + noise model (always True when flagged and not checked)
    s5: float               #: level used to re-degrade the source (plug-ins, phi)
    sigma5: float           #: noise used to re-degrade the source
    s_joint: float          #: v1.0 joint blur + noise fit (reported, superseded by v2)
    sigma_joint: float
    gamma_med: float        #: median knee exponent of the target
    knee_med: float         #: median raw knee of the target
    n: int                  #: number of target images

    def as_dict(self) -> dict:
        return asdict(self)

    def notebook_names(self) -> dict:
        """The same quantities under the column names of the notebook's tables."""
        return dict(s_hat=self.s_hat, p_gof=self.p_fit, flagged=self.flagged, s_v11=self.s_tail, sig_v11=self.sigma_tail,
                    p_gof2=self.p_noise, flagged2=self.noise_rejected, s5=self.s5, sigma5=self.sigma5, s_use=self.s_joint,
                    sig_use=self.sigma_joint, gamma_med=self.gamma_med, knee_med=self.knee_med, n=self.n)


class BlurIdentifier:
    """Calibrate on labelled source images, then identify the blur of unlabelled target sets.

    >>> ident = BlurIdentifier().fit(source_tiles)        # uint8 (n, 224, 224, 3), labels not needed
    >>> res = ident.identify(target_tiles)                # Identification(s_hat=..., flagged=..., s5=..., sigma5=...)
    """

    def __init__(self, joint_fit: bool = True):
        self.joint_fit = joint_fit
        self.tables: PixelTables | None = None
        self.src_tiles = None
        self.W_src = None
        self.cal: Calibration | None = None

    def fit(self, source_tiles: np.ndarray) -> BlurIdentifier:
        src = np.asarray(source_tiles)
        if src.ndim != 4:
            raise ValueError("source tiles must have shape (n, H, W, channels)")
        self.tables = tables_for(src.shape[1], src.shape[2])
        self.src_tiles = src
        self.W_src = spectra(src, tables=self.tables)
        self.cal = calibrate(self.W_src, self.tables, joint=self.joint_fit)
        return self

    def identify(self, target_tiles: np.ndarray | None = None, *, W: np.ndarray | None = None, noise_check: bool = True,
                 key=("target",)) -> Identification:
        """Identify the blur of one target set (tiles, or their spectra ``W``). ``key`` names the random streams of the
        noise-model check, so that a result can be recomputed exactly."""
        if self.cal is None:
            raise RuntimeError("call fit(source_tiles) first")
        if W is None:
            tgt = np.asarray(target_tiles)
            if tgt.shape[1:3] != (self.tables.h, self.tables.w):
                raise ValueError(f"target tiles are {tgt.shape[1:3]}, the source was {self.tables.h}x{self.tables.w}")
            W = spectra(tgt, tables=self.tables)
        o, _, res = identify_spectra(W, self.W_src, self.cal, self.tables)
        if o["flagged"] and np.isfinite(o["s_v11"]) and noise_check:
            o["p_gof2"] = noise_model_pvalue(self.src_tiles, res, o["s_v11"], o["sig_v11"], key, tables=self.tables)
            o["flagged2"] = bool(o["p_gof2"] < GOF_ALPHA)
        else:
            o["p_gof2"], o["flagged2"] = (o["p_gof"], False) if not o["flagged"] else (np.nan, True)
        s5, g5 = plugin_level(o)
        return Identification(s_hat=o["s_hat"], p_fit=o["p_gof"], flagged=o["flagged"], s_tail=o["s_v11"],
                              sigma_tail=o["sig_v11"], p_noise=o["p_gof2"], noise_rejected=o["flagged2"], s5=s5, sigma5=g5,
                              s_joint=o["s_use"], sigma_joint=o["sig_use"], gamma_med=o["gamma_med"], knee_med=o["knee_med"],
                              n=o["n"])
