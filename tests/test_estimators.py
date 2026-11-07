"""Published estimators, temperature scaling, plug-ins, attribution and the floor-sensitivity index."""
import math

import numpy as np
import pytest

from redeg import (
    attribution_score,
    bias_transfer,
    blend,
    fit_temperature,
    floor_sensitivity_index,
    labelled_share,
    plugins_at_level,
    predicts_harm,
    softmax,
    softmax_T,
    switch,
)
from redeg import estimators as E
from redeg._keys import rng_for


@pytest.fixture(scope="module")
def logits():
    r = rng_for("test_logits")
    Zs, Zt = r.standard_normal((400, 6)) * 2.5, r.standard_normal((400, 6)) * 1.5
    ys = (Zs + r.standard_normal((400, 6))).argmax(1)
    return Zs, Zt, ys


def test_atc_threshold_semantics():
    src = np.linspace(0, 1, 101)
    # source accuracy 0.7 -> threshold at the 30% quantile -> 70% of the source lies above it
    assert E.atc_predict(src, src, 0.7) == pytest.approx(0.7, abs=0.011)
    assert E.atc_predict(src - 1.0, src, 0.7) == 0.0


def test_doc_and_ac(logits):
    Zs, Zt, ys = logits
    Ps, Pt = softmax(Zs), softmax(Zt)
    acc = float((Ps.argmax(1) == ys).mean())
    assert E.doc(Ps, Ps, acc) == pytest.approx(acc)
    assert E.ac(Pt) == pytest.approx(Pt.max(1).mean())


def test_temperature_recovers_a_known_scale():
    r = rng_for("test_temperature")
    Z = r.standard_normal((4000, 10)) * 3.0
    y = np.array([r.choice(10, p=p) for p in softmax(Z)])     # labels drawn from softmax(Z): calibrated at T = 1
    T, nll1, nllT = fit_temperature(2.0 * Z, y)               # logits twice too large -> T close to 2
    assert abs(T - 2.0) < 0.15 and nllT <= nll1
    assert np.allclose(softmax_T(2.0 * Z, T).sum(1), 1.0)


def test_mano_score_ranges(logits):
    Zs, Zt, _ = logits
    for soft in (True, False):
        s = E.mano_score(Zt, soft)
        assert 0 < s <= 1
    assert E.mano_phi(np.zeros((5, 4))) == pytest.approx(math.log(4))


def test_nuclear_norm_score_bounds(logits):
    _, Zt, _ = logits
    s = E.nuc_score(softmax(Zt))
    assert 0 < s <= 1
    assert E.nuc_score(np.eye(6)[np.arange(60) % 6]) == pytest.approx(1.0)


def test_cot_transport_equals_assignment():
    """The C x n transportation of COT equals the n x n assignment (the assertion of Cell 4.2)."""
    pytest.importorskip("ot")
    from scipy.optimize import linear_sum_assignment

    for t in range(5):
        r = rng_for("cot_equivalence", t)
        n, C = 14, 4
        P = r.dirichlet(np.ones(C) * 0.7, size=n)
        cnt = E.sample_label_counts(np.full(C, 1.0 / C), n, r)
        labels = np.repeat(np.arange(C), cnt)
        cost = 0.5 * np.abs(np.eye(C)[labels][:, None, :] - P[None, :, :]).sum(-1)
        ri, ci = linear_sum_assignment(cost)
        assert abs(cost[ri, ci].mean() - E.ot_costs(P, cnt, False).mean()) < 1e-9


def test_plugins_are_identities_at_level_zero(logits):
    """Every plug-in returns the published estimator when the re-degraded source is the clean source (rule L1 / Q1)."""
    pytest.importorskip("ot")
    Zs, Zt, ys = logits
    Ps, Pt = softmax(Zs), softmax(Zt)
    out = plugins_at_level(Pt, Ps, ys, Ps, level_key=("lvl", 0.0, 0.0))
    for k in ("ATC", "ATC_NE", "DoC", "COT", "COTT"):
        assert out[k + "@"] == out[k]
    lv = plugins_at_level(Pt, softmax(0.6 * Zs), ys, Ps, level_key=("lvl", 4.0, 0.0))
    assert lv["COT"] == out["COT"] and lv["ATC"] == out["ATC"]


def test_bias_transfer_identity():
    assert bias_transfer(0.73, 0.02, 0.02) == 0.73
    assert bias_transfer(0.73, 0.10, 0.02) == pytest.approx(0.65)
    assert bias_transfer(1.2, 0.0, 0.0) == 1.0


def test_attribution_score_identities():
    assert attribution_score(0.9, 0.7, 0.7) == 1.0               # re-degraded source closes the whole gap
    assert attribution_score(0.9, 0.9, 0.7) == 0.0               # s_hat = 0: nothing is explained
    assert np.isnan(attribution_score(0.9, 0.8, 0.897))          # gap below 0.005: undefined
    assert labelled_share(0.8, 0.7, 0.6) == pytest.approx(0.5)
    assert np.isnan(labelled_share(0.8, 0.79, 0.79))             # labelled drop below 2 pp


def test_switch_and_blend():
    assert switch(0.6, 0.8, 0.5) == 0.8 and switch(0.6, 0.8, 0.49) == 0.6 and switch(0.6, 0.8, np.nan) == 0.6
    assert blend(0.6, 0.8, 0.5) == pytest.approx(0.7) and blend(0.6, 0.8, 2.0) == pytest.approx(0.8)
    assert blend(0.6, np.nan, 0.5) == 0.6


def test_floor_sensitivity_index_by_hand():
    r = rng_for("test_fsi")
    y = r.integers(0, 5, 200)
    Zc = {s: r.standard_normal((200, 5)) for s in (1.0, 4.0, 16.0)}
    Zn = {s: r.standard_normal((200, 5)) * 2 for s in (1.0, 4.0, 16.0)}
    Cc = {s: Z.argmax(1) == y for s, Z in Zc.items()}
    Cn = {s: Z.argmax(1) == y for s, Z in Zn.items()}
    f = floor_sensitivity_index(Zc, Cc, Zn, Cn, T=1.3)
    manual = np.mean([100 * ((softmax_T(Zn[s], 1.3).max(1).mean() - Cn[s].mean())
                             - (softmax_T(Zc[s], 1.3).max(1).mean() - Cc[s].mean())) for s in (1.0, 4.0, 16.0)])
    assert f["fsi"] == pytest.approx(manual, abs=1e-12)
    assert predicts_harm(4.0) and not predicts_harm(3.99)
