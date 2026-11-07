"""Identification of the blur level without labels (v1.0) and the noise-aware path (v2) on synthetic tiles."""
import numpy as np
import pytest

from redeg import BlurIdentifier, plugin_level, spectra
from redeg._keys import rng_for
from redeg.semigroup import dct2, spectra_from_coeffs, tables_for


@pytest.fixture(scope="module")
def ident(source_tiles):
    return BlurIdentifier().fit(source_tiles)


def _blurred_spectra(tiles, s0, sigma=0.0, key="noise"):
    C = dct2(tiles.astype(np.float64).transpose(0, 3, 1, 2)) * np.exp(-0.5 * s0 * tables_for(224).lam)
    if sigma:
        C = C + sigma * rng_for("test_" + key, s0, sigma).standard_normal(C.shape)
    return spectra_from_coeffs(C)


@pytest.mark.parametrize("s0", [2.0, 8.0, 32.0])
def test_clean_blur_is_identified_in_px2(ident, target_tiles, s0):
    res = ident.identify(W=_blurred_spectra(target_tiles, s0))
    assert not res.flagged
    assert abs(res.s_hat - s0) / s0 <= 0.15                     # rule F4 tolerance
    assert (res.s5, res.sigma5) == (res.s_hat, 0.0)


def test_unblurred_target_gives_a_small_level(ident, target_tiles):
    res = ident.identify(target_tiles)
    assert res.s_hat < 0.5 and not res.flagged


def test_noise_is_flagged_and_the_v2_path_recovers_blur_and_noise(ident, target_tiles):
    res = ident.identify(W=_blurred_spectra(target_tiles, 6.0, sigma=6.0), key=("test",))
    assert res.flagged                                          # blur alone does not explain the target
    assert abs(res.s_tail - 6.0) / 6.0 <= 0.25
    assert abs(res.sigma_tail - 6.0) / 6.0 <= 0.2
    assert not res.noise_rejected                               # the blur + noise model fits
    assert (res.s5, res.sigma5) == (res.s_tail, res.sigma_tail)


def test_identify_from_tiles_equals_identify_from_spectra(ident, target_tiles):
    a = ident.identify(target_tiles[:8])
    b = ident.identify(W=spectra(target_tiles[:8]))
    assert a.as_dict() == b.as_dict()


def test_plugin_level_rules():
    assert plugin_level(dict(flagged=False, s_use=3.0, s_v11=9.0, sig_v11=1.0)) == (3.0, 0.0)
    assert plugin_level(dict(flagged=True, s_use=1.0, s_v11=5.5, sig_v11=2.5, flagged2=False)) == (5.5, 2.5)
    assert plugin_level(dict(flagged=True, s_use=1.0, s_v11=5.5, sig_v11=2.5, flagged2=True)) == (5.5, 0.0)
    assert plugin_level(dict(flagged=True, s_use=1.0, s_v11=5.5, sig_v11=2.5)) == (5.5, 0.0)


def test_size_mismatch_is_an_error(ident):
    with pytest.raises(ValueError):
        ident.identify(np.zeros((2, 32, 32, 3), np.uint8))
