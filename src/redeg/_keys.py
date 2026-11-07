"""Keyed random streams (notebook Cell 1.4).

Every random draw of the experiments comes from a stream named by a key, never from a global seed, so that a quantity
can be recomputed in any order and in any session. The library uses the same construction.
"""
from __future__ import annotations

import hashlib

import numpy as np

GLOBAL_SEED = 0


def stable_hash(key) -> int:
    """First 8 hex digits of ``md5(str(key))`` as an integer (never Python's salted ``hash``)."""
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest()[:8], 16)


def rng_for(*keys) -> np.random.Generator:
    """The NumPy generator of the stream named ``keys``."""
    return np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, *[stable_hash(k) for k in keys]]))


def seed_for(*keys) -> int:
    """An integer seed drawn from the stream named ``keys`` (used to seed other generators)."""
    return int(rng_for(*keys).integers(0, 2 ** 62))
