"""Shared fixtures: keyed synthetic power-law tiles (natural-image-like, |omega|^-2 spectra) in DN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from redeg._keys import rng_for  # noqa: E402
from redeg.semigroup import idct2, tables_for  # noqa: E402
from redeg.theory import powerlaw_coeffs  # noqa: E402


def make_tiles(n: int, key: str, alpha: float = 2.0, pre_blur: float = 0.7, size: int = 224) -> np.ndarray:
    """uint8 (n, size, size, 3) tiles of isotropic |omega|^-alpha fields, slightly pre-blurred, quantised to 8 bits."""
    tab = tables_for(size)
    c = powerlaw_coeffs(alpha, n, rng_for(key), channels=3, tables=tab) * np.exp(-0.5 * pre_blur * tab.lam)
    return np.clip(np.round(idct2(c) * 25.0 + 120.0), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


@pytest.fixture(scope="session")
def source_tiles() -> np.ndarray:
    return make_tiles(24, "test_source")


@pytest.fixture(scope="session")
def target_tiles() -> np.ndarray:
    return make_tiles(16, "test_target")
