"""Theorem 2 and Corollary 1 on synthetic power-law fields; reproduction of block B1 of the final run."""
import numpy as np

from redeg import drho_ds0, half_point, rho_curves
from redeg._keys import rng_for
from redeg.response import drho_ds0_fd
from redeg.semigroup import spectra, spectra_from_coeffs, tables_for
from redeg.theory import gamma_from_alpha, half_energy_point, knee_law, knee_law_error_bound, log_sensitivity, powerlaw_coeffs


def test_block_b1_half_energy_law_is_reproduced_exactly():
    """eta_1/2 / s0 (median over 16 fields and s0 in {2, 4, 8, 16}) from the keyed streams of block B1.

    Published (run log of the final run; paper Sec. 6.1): 0.720, 0.961, 1.489, 3.416 for alpha = 1.5, 2, 2.5, 3.
    """
    etas = np.geomspace(0.01, 1e4, 600)
    got = []
    for alpha in (1.5, 2.0, 2.5, 3.0):
        W0 = spectra_from_coeffs(powerlaw_coeffs(alpha, 16, rng_for("powerlaw", alpha)))
        rat = [half_point(R, etas) / s0 for s0 in (2.0, 4.0, 8.0, 16.0) for R in rho_curves(W0, s0, etas)]
        got.append(np.median(rat))
    assert np.round(got, 3).tolist() == [0.72, 0.961, 1.489, 3.416]
    pred = [half_energy_point(1.0, gamma_from_alpha(a)) for a in (1.5, 2.0, 2.5, 3.0)]
    assert np.allclose(pred, [2 ** (1 / 1.25) - 1, 1.0, 2 ** (1 / 0.75) - 1, 3.0])


def test_response_is_bounded_and_gain_invariant(source_tiles):
    x = source_tiles[:4].astype(np.float64)
    R1 = rho_curves(spectra(x))
    R2 = rho_curves(spectra(3.0 * x - 40.0))
    assert R1.min() >= 0 and R1.max() <= 1
    assert np.abs(R1 - R2).max() <= 1e-12


def test_sensitivity_identity_matches_finite_difference():
    """Rule R1: analytic d rho / d s0 (Theorem 2b) against a central finite difference (median relative error <= 1e-3)."""
    lam = tables_for(224).lam
    W = spectra_from_coeffs(powerlaw_coeffs(2.0, 6, rng_for("r1_preview"), channels=3) * np.exp(-0.5 * 0.6 * lam))
    etas = np.geomspace(0.25, 1024.0, 13)
    an, fd = drho_ds0(W, etas), drho_ds0_fd(W, etas)
    assert np.median(np.abs(an - fd) / np.abs(an)) <= 1e-3


def test_knee_law_bound_holds_on_power_law_fields():
    """Theorem 2(e) for alpha = 2 (gamma = 1): rho is within the incomplete-gamma bound of the knee law."""
    W = spectra_from_coeffs(powerlaw_coeffs(2.0, 8, rng_for("knee_bound")))
    etas = np.geomspace(0.25, 1024.0, 13)
    for s0 in (2.0, 8.0):
        R = rho_curves(W, s0, etas).mean(0)
        dev = np.abs(R - knee_law(etas, s0, 1.0))
        # the fields are random draws of the power law, so allow a sampling margin on top of the bound
        assert np.all(dev <= knee_law_error_bound(etas, s0, 1.0, 8.0) + 0.05)


def test_sensitivity_peaks_at_one_over_gamma():
    for gamma in (0.75, 1.0, 1.5):
        u = np.geomspace(1e-3, 1e3, 20001)
        assert abs(u[np.argmax(log_sensitivity(u, gamma))] - 1 / gamma) / (1 / gamma) < 1e-3
