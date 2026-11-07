"""The package against outputs of the final notebook's own code (``tests/fixtures/notebook_reference.npz``).

The fixture was written by ``scripts/check_notebook_equivalence.py --notebook <final notebook> --write-fixture ...``, which
executes the notebook's definitions (Cells 1.x-6.4m) on keyed synthetic inputs. On the machine that wrote it every
quantity is bit-identical; here a relative tolerance of 1e-9 absorbs last-bit differences between BLAS / FFT builds.
"""
from pathlib import Path

import numpy as np
import pytest
from check_notebook_equivalence import compare, make_inputs, package_outputs

FIXTURE = Path(__file__).parent / "fixtures" / "notebook_reference.npz"


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# quantities that depend on the build of a third-party library rather than on this code: checked on the machine that wrote
# the fixture (bit-identical), and here only for the decision they feed (p-values) or not at all (libjpeg decoding)
PVALUES = ("id_clean_p_gof", "id_noisy_p_gof", "id_noisy_p2")
LIBRARY_BUILD = ("jpeg_rt", "temp") + PVALUES


def test_package_reproduces_the_notebook(reference):
    try:
        import ot  # noqa: F401
    except ImportError:
        reference = {k: v for k, v in reference.items() if k not in ("plugins_level", "plugins_zero", "baselines")}
    pkg = package_outputs(make_inputs(), T=float(reference["temp"][0]))
    strict = {k: v for k, v in reference.items() if k not in LIBRARY_BUILD}
    bad = compare(strict, pkg, rtol=1e-9, atol=1e-12, verbose=False)
    assert not bad, f"differs from the notebook: {bad}"
    assert np.allclose(pkg["temp"], reference["temp"], rtol=1e-4)
    for k in PVALUES:
        assert (float(pkg[k]) < 0.01) == (float(reference[k]) < 0.01), k


def test_fixture_contains_the_published_half_energy_table(reference):
    assert np.round(reference["b1_half_energy"], 3).tolist() == [0.72, 0.961, 1.489, 3.416]


def test_fixture_reproduces_the_b27_registration_hash(reference):
    assert str(reference["b27_sha256"]) == "e7c59048e412a5a376e40d5fbc241323e264ba02abfd1979bc09c583bd57078b"
