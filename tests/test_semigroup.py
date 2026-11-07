"""Lemma 1 and the exactness checks of rule R0 / R2 (paper Sec. 3; notebook Cell 2.2) on the package's operators."""
import numpy as np
import pytest

from redeg import heat, standardize
from redeg._keys import rng_for
from redeg.semigroup import dct2, dct_matrix, idct2, spectra, tables_for

TOL = 1e-10


@pytest.fixture(scope="module")
def fields():
    r = rng_for("test_fields")
    return r.standard_normal((3, 224, 224)) * 30 + 100


def test_semigroup_property(fields):
    for x in fields:
        assert np.abs(heat(heat(x, 1.3), 2.1) - heat(x, 3.4)).max() <= TOL * np.abs(x).max()


def test_parseval_gradient_energy(fields):
    """||D x||^2 with non-wrapping forward differences equals sum_lambda lambda c^2."""
    lam = tables_for(224).lam
    for x in fields:
        dx = np.sum(np.diff(x, axis=0) ** 2) + np.sum(np.diff(x, axis=1) ** 2)
        e = np.sum(lam * dct2(x) ** 2)
        assert abs(dx - e) / e <= TOL


def test_constants_are_invariant():
    assert np.abs(heat(np.full((224, 224), 3.7), 5.0) - 3.7).max() <= TOL


def test_gain_offset_commute_and_standardisation(fields):
    """T_s A = A T_s and N o A = N (Lemma 1)."""
    r = rng_for("test_gain")
    for x in fields:
        for s in (0.5, 4.0, 64.0):
            a, b = float(r.uniform(0.2, 5.0)), float(r.uniform(-50, 50))
            assert np.abs(heat(a * x + b, s) - (a * heat(x, s) + b)).max() <= TOL * (a * np.abs(x).max() + abs(b))
            assert np.abs(standardize(a * x + b) - standardize(x)).max() <= TOL


def test_fast_dct_matches_matrix_reference(fields):
    D = dct_matrix(224)
    for x in fields:
        assert np.abs(dct2(x) - D @ x @ D.T).max() <= 1e-9 * np.abs(D @ x @ D.T).max()
        assert np.abs(idct2(dct2(x)) - x).max() <= 1e-9 * np.abs(x).max()


def test_narrowband_obstruction():
    """Rule R2: a single DCT mode is an eigenfunction, so N T_s x does not depend on s (Theorem 2d)."""
    D = dct_matrix(224)
    for k, l in ((1, 0), (3, 2), (10, 5)):
        x = 10 + np.outer(D[k], D[l]) * 224
        ref = standardize(x)
        for s in (0.5, 2.0, 8.0, 32.0):
            assert np.abs(standardize(heat(x, s)) - ref).max() <= TOL


def test_spectra_shape_and_total_energy(source_tiles):
    W = spectra(source_tiles[:4])
    assert W.shape == (4, 3, 3000) and W.dtype == np.float64
    x = source_tiles[0].astype(np.float64).transpose(2, 0, 1)
    lam = tables_for(224).lam
    assert np.allclose(W[0].sum(-1), (lam * dct2(x) ** 2).sum((-2, -1)), rtol=1e-12)


def test_other_sizes_use_their_own_tables():
    t32 = tables_for(32)
    assert t32.lam.shape == (32, 32) and t32.nbins == 3000
    x = rng_for("test_32").standard_normal((2, 3, 32, 32))
    assert np.abs(heat(heat(x, 0.5), 0.25) - heat(x, 0.75)).max() <= TOL
