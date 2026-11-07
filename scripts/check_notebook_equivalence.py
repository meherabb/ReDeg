#!/usr/bin/env python3
"""Check that the ``redeg`` package computes exactly what the notebook computed.

The script reads the final notebook (``notebooks/EXP9/EXP9_v8_final.ipynb``) as JSON, extracts by AST only the top-level
definitions the method needs (functions and constants of Cells 1.x, 2.1, 2.3, 3.1-3.3, 4.1, 4.2, 4.4, 5.11, 6.4c, 6.4g,
6.4k and 6.4m), executes them in a fresh namespace, and compares them with the package on keyed synthetic inputs. No data
are downloaded and no notebook cell is run as a whole.

    python scripts/check_notebook_equivalence.py --notebook notebooks/EXP9/EXP9_v8_final.ipynb
    python scripts/check_notebook_equivalence.py --notebook ... --write-fixture tests/fixtures/notebook_reference.npz
    python scripts/check_notebook_equivalence.py --fixture tests/fixtures/notebook_reference.npz

The fixture stores the notebook's outputs; ``tests/test_notebook_reference.py`` checks the package against it without the
notebook. Needs numpy, scipy, pillow and POT.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# cell header prefix -> names taken from that cell (the first top-level definition of each name, in cell order)
SPEC = [
    ("1.", ["GLOBAL_SEED", "stable_hash", "rng_for", "seed_for"]),
    ("2.1", ["TILE", "_k", "DCT", "LAM1", "LAM", "S_ACC", "ETA_PIX", "NOISE_DN", "dct2_ref", "idct2_ref", "dct2", "idct2", "heat",
             "N_exact", "N_eps", "_LF", "_POS", "EDGES", "NBINS", "_BIN", "_CNT", "LAMB", "_BINMAT_T", "spectra_from_coeffs",
             "spectra", "rho_curves", "drho_ds0", "drho_ds0_fd", "_trapz", "observability", "narrowband_index", "powerlaw_coeffs"]),
    ("2.3", ["ALPHAS", "S0_PL", "ETA_FINE", "half_point_curve"]),
    ("3.1", ["S_FIT", "G_FIT", "R_FIT", "CAL_GRID", "GOF_ALPHA", "_BASE", "_MOD", "EW", "_HI", "IDX_ETA4", "knee_fit_ref",
             "knee_fit_noise_ref", "_GRID", "_M1", "_MN", "_MNSQ", "knee_fit", "knee_fit_noise"]),
    ("3.2", ["inv", "inv_decreasing", "noise_sigma", "build_cal", "S2", "G2", "_X2", "_NB", "_tail_prep", "tail_fit_ref", "tail_fit",
             "identify_v11", "identify", "noise_model_pvalue"]),
    ("3.3", ["maxconf", "negent", "atc_predict"]),
    ("4.1", ["entropy", "nuc_score", "clip01"]),
    ("4.2", ["sample_label_counts", "_ot_costs", "cott_threshold"]),
    ("4.4", ["_cot_cost", "plugin_family", "_cot_bias", "plugin_family_v5"]),
    ("5.11", ["_QTAB", "_pillow_qtables", "jpeg_file_store", "read_store", "store_label", "jpeg_roundtrip7"]),
    ("6.4c", ["_softmax"]),
    ("6.4g", ["MANO_P", "MANO_ETA", "PHI_MIN_GAP", "PHI_SWITCH", "softmax_T", "fit_temperature", "mano_phi", "mano_score"]),
    ("6.4k", ["FSI_LEVELS", "v5_level", "_st_exact"]),
    ("6.4m", ["fsi_from_logits", "_canonical_registration"]),
]
N_SRC, N_TGT, K_CLS, N_LOGIT = 16, 8, 7, 300


# ------------------------------------------------------------------ extraction
def code_cells(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", ""))
            m = re.match(r"\s*# === ([0-9]+(?:\.[0-9]+[a-z]?)?)", src)
            yield (m.group(1) if m else ""), src


def defined_names(node) -> set:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        t = node.target
        while isinstance(t, (ast.Subscript, ast.Attribute)):
            t = t.value
        return {t.id} if isinstance(t, ast.Name) else set()
    return set()


def build_namespace(nb_path: Path) -> dict:
    import scipy.fft as sfft
    import scipy.sparse as sps
    from PIL import Image, JpegImagePlugin
    from scipy.optimize import minimize_scalar
    from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon

    ns = dict(np=np, sfft=sfft, sps=sps, mannwhitneyu=mannwhitneyu, spearmanr=spearmanr, wilcoxon=wilcoxon, rankdata=rankdata,
              math=math, io=io, json=json, hashlib=hashlib, Counter=Counter, Image=Image, JpegImagePlugin=JpegImagePlugin,
              minimize_scalar=minimize_scalar, CORR_THREADS=1,
              PREREG=json.loads((ROOT / "prereg" / "v1.0.json").read_text(encoding="utf-8")))
    try:
        import ot
        ns["ot"] = ot
    except ImportError:
        pass
    cells = list(code_cells(nb_path))
    found: set = set()
    for prefix, names in SPEC:
        want = set(names)
        for head, src in cells:
            if not (head == prefix or (prefix.endswith(".") and head.startswith(prefix))):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in tree.body:
                d = defined_names(node) & want
                if not d or (d <= found and not isinstance(node, ast.AugAssign)):
                    continue
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"<cell {head}>", "exec"), ns)
                found |= d
        missing = want - found
        if missing:
            raise SystemExit(f"cell {prefix}: definitions not found in the notebook: {sorted(missing)}")
    return ns


# ------------------------------------------------------------------ inputs (keyed streams, identical for both sides)
def make_inputs() -> dict:
    from redeg._keys import rng_for
    from redeg.semigroup import idct2, tables_for
    from redeg.theory import powerlaw_coeffs

    lam = tables_for(224).lam
    c = powerlaw_coeffs(2.0, N_SRC + N_TGT, rng_for("equiv_tiles"), channels=3) * np.exp(-0.5 * 0.7 * lam)
    tiles = np.clip(np.round(idct2(c) * 25.0 + 120.0), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    r = rng_for("equiv_logits")
    Zs, Zt = r.standard_normal((N_LOGIT, K_CLS)) * 2.5, r.standard_normal((N_LOGIT, K_CLS)) * 1.6
    ys = (Zs + r.standard_normal((N_LOGIT, K_CLS))).argmax(1)
    Zc = np.stack([r.standard_normal((N_LOGIT, K_CLS)).astype(np.float32) * (2.0 / (1 + s)) for s in (1.0, 4.0, 16.0)])
    Zn = np.stack([r.standard_normal((N_LOGIT, K_CLS)).astype(np.float32) * (1.5 / (1 + s)) for s in (1.0, 4.0, 16.0)])
    return dict(tiles=tiles, Zs=Zs, Zt=Zt, Zl=Zs * 0.7, ys=ys, fsi_Zc=Zc, fsi_Zn=Zn)


def _noisy_coeffs(tiles, s, sigma, key):
    """DCT coefficients of tiles blurred by s plus white noise sigma (exact in the DCT domain), keyed."""
    from redeg._keys import rng_for
    from redeg.semigroup import dct2, tables_for

    C = dct2(tiles.astype(np.float64).transpose(0, 3, 1, 2)) * np.exp(-0.5 * s * tables_for(224).lam)
    return C + sigma * rng_for(*key).standard_normal(C.shape)


def _sub(a):
    """A fixed subsample of a large array (stored instead of the whole array)."""
    a = np.asarray(a)
    return a.ravel()[:: max(1, a.size // 4000)]


# ------------------------------------------------------------------ the two sides
def notebook_outputs(nb: dict, inp: dict) -> dict:
    tiles, out = inp["tiles"], {}
    out["rng_check"] = nb["rng_for"]("equiv_check", 1.5).standard_normal(6)
    for k in ("LAM", "LAMB", "_CNT", "EDGES", "_BIN", "EW", "_HI", "_NB"):
        out["tab" + k] = _sub(nb[k])
    out["heat"] = _sub(nb["heat"](tiles[:2].astype(np.float64).transpose(0, 3, 1, 2), 3.7))
    W = nb["spectra"](tiles)
    out["W"] = _sub(W)
    R0 = nb["rho_curves"](W, 0.0)
    out["rho_0"], out["rho_5"] = R0, nb["rho_curves"](W, 5.0)
    out["drho"], out["obs"], out["nbi"] = nb["drho_ds0"](W, nb["ETA_PIX"]), nb["observability"](W), nb["narrowband_index"](W)
    for i, v in enumerate(nb["knee_fit"](R0)):
        out[f"knee_{i}"] = v
    for i, v in enumerate(nb["knee_fit_noise"](R0)):
        out[f"kneen_{i}"] = v
    Wn = nb["spectra_from_coeffs"](_noisy_coeffs(tiles, 0.0, 4.0, ("equiv_noise",)))
    for i, v in enumerate(nb["tail_fit"](Wn)):
        out[f"tail_{i}"] = v
    W_src = W[:N_SRC]
    cal = nb["build_cal"](W_src)
    out["cal1"], out["cal2"] = cal["CAL1"], cal["CAL2"]
    targets = {"clean": W[N_SRC:] * np.exp(-6.0 * nb["LAMB"])[None, None, :],
               "noisy": nb["spectra_from_coeffs"](_noisy_coeffs(tiles[N_SRC:], 6.0, 6.0, ("equiv_tnoise",)))}
    for name, Wt in targets.items():
        o, ks, res = nb["identify"](Wt, W_src, cal, variants=True)
        for k in ("s_hat", "p_gof", "flagged", "s_use", "sig_use", "s_v11", "sig_v11", "gamma_med", "knee_med"):
            out[f"id_{name}_{k}"] = np.asarray(float(o[k]))
        out[f"id_{name}_res"] = res
        if o["flagged"] and np.isfinite(o["s_v11"]):
            out[f"id_{name}_p2"] = np.asarray(nb["noise_model_pvalue"](tiles[:N_SRC], res, o["s_v11"], o["sig_v11"], ("equiv", name)))
    rows = []
    for alpha in nb["ALPHAS"]:
        W0 = nb["spectra_from_coeffs"](nb["powerlaw_coeffs"](alpha, 16, nb["rng_for"]("powerlaw", alpha)))
        rat = []
        for s0 in nb["S0_PL"]:
            R = nb["rho_curves"](W0, s0, nb["ETA_FINE"])
            rat += [nb["half_point_curve"](R[i], nb["ETA_FINE"]) / s0 for i in range(16)]
        rows.append(np.median(rat))
    out["b1_half_energy"] = np.array(rows)
    Zs, Zt, Zl, ys = inp["Zs"], inp["Zt"], inp["Zl"], inp["ys"]
    n = len(ys)
    T, nll1, nllT = nb["fit_temperature"](Zs, ys, np.ones(n, bool))
    out["temp"] = np.array([T, nll1, nllT])
    Ps, Pt, Pl = (nb["softmax_T"](Z, T) for Z in (Zs, Zt, Zl))
    out["negent"], out["nuc"] = nb["negent"](Pt), np.asarray(nb["nuc_score"](Pt))
    out["mano"] = np.array([nb["mano_score"](Zt / T, True), nb["mano_score"](Zt / T, False), nb["mano_phi"](Zs / T)])
    if "ot" in nb:
        prior = np.bincount(ys, minlength=Zs.shape[1]) / n
        c0 = Ps.argmax(1) == ys
        ctx = dict(level_cache={}, valid=np.ones(n, bool), ycol=ys, prior=prior, src_probs=lambda s, g, tag: (Ps, c0))
        cost_t = nb["_cot_cost"](Pt, prior, nb["rng_for"]("equiv_target", "cot"))
        cott_t = nb["_ot_costs"](Pt, nb["sample_label_counts"](prior, n, nb["rng_for"]("equiv_target", "cott")), True)
        fam = nb["plugin_family_v5"](ctx, Pt, Pl, Pl.argmax(1) == ys, ("lvl", 4.0, 0.0), cost_t, cott_t)
        fam0 = nb["plugin_family_v5"](ctx, Pt, Ps, c0, ("lvl", 0.0, 0.0), cost_t, cott_t)
        acc0 = float(c0.mean())
        tau0 = nb["cott_threshold"](Ps, ys)
        keys = ("ATC", "ATC_NE", "DoC", "COT", "COTT")
        out["plugins_level"] = np.array([fam[k] for k in keys])
        out["plugins_zero"] = np.array([fam0[k] for k in keys])
        out["baselines"] = np.array([nb["atc_predict"](nb["maxconf"](Pt), nb["maxconf"](Ps), acc0),
                                     nb["atc_predict"](nb["negent"](Pt), nb["negent"](Ps), acc0),
                                     nb["clip01"](acc0 - (nb["maxconf"](Ps).mean() - nb["maxconf"](Pt).mean())),
                                     nb["clip01"](1.0 - cost_t), nb["clip01"](1.0 - float(np.mean(cott_t > tau0)))])
    rows = []
    for q in (50, 85, 95):
        for sub in (0, 1, 2):
            buf = io.BytesIO()
            nb["Image"].fromarray(tiles[0]).save(buf, format="JPEG", quality=q, subsampling=sub)
            st = nb["jpeg_file_store"](buf.getvalue())
            rows.append([q, sub, st[0], st[1], int(st[2])])
    out["store_rows"] = np.array(rows)
    out["jpeg_rt"] = _sub(nb["jpeg_roundtrip7"](tiles[:4], 85, 2))
    lv = nb["FSI_LEVELS"]
    Zc = {float(s): inp["fsi_Zc"][i] for i, s in enumerate(lv)}
    Zn = {float(s): inp["fsi_Zn"][i] for i, s in enumerate(lv)}
    f = nb["fsi_from_logits"](Zc, {s: Z.argmax(1) == ys for s, Z in Zc.items()}, Zn, {s: Z.argmax(1) == ys for s, Z in Zn.items()}, 0.9)
    out["fsi"] = np.array([f["fsi"]] + [f["per_level"][float(s)] for s in lv])
    rec = json.loads((ROOT / "prereg" / "b27_registration_record.json").read_text(encoding="utf-8"))
    out["b27_sha256"] = np.asarray(hashlib.sha256(nb["_canonical_registration"](rec["models"]).encode("utf-8")).hexdigest())
    out["v5_level"] = np.array([nb["v5_level"](d) for d in _V5_CASES])
    return out


_V5_CASES = [dict(flagged=False, s_use=3.0, s_v11=3.0, sig_v11=0.0, flagged2=False),
             dict(flagged=True, s_use=2.0, s_v11=5.5, sig_v11=2.5, flagged2=False),
             dict(flagged=True, s_use=2.0, s_v11=5.5, sig_v11=2.5, flagged2=True),
             dict(flagged=True, s_use=2.0, s_v11=5.5, sig_v11=2.5)]


def package_outputs(inp: dict, T: float | None = None) -> dict:
    """``T`` fixes the temperature used downstream of the fit (the unit test passes the notebook's, so that a different
    SciPy optimiser build cannot propagate into the softmax-based quantities)."""
    import redeg
    from redeg import estimators as E
    from redeg._keys import rng_for
    from redeg.semigroup import spectra_from_coeffs
    from redeg.storage import jpeg_file_store, jpeg_roundtrip
    from redeg.theory import powerlaw_coeffs

    tiles, out = inp["tiles"], {}
    out["rng_check"] = rng_for("equiv_check", 1.5).standard_normal(6)
    tab = redeg.tables_for(224)
    for k, v in (("LAM", tab.lam), ("LAMB", tab.lamb), ("_CNT", tab.counts), ("EDGES", tab.edges), ("_BIN", tab.bin_index),
                 ("EW", tab.ew), ("_HI", tab.hi), ("_NB", tab.noise_energy)):
        out["tab" + k] = _sub(v)
    out["heat"] = _sub(redeg.heat(tiles[:2].astype(np.float64).transpose(0, 3, 1, 2), 3.7))
    W = redeg.spectra(tiles)
    out["W"] = _sub(W)
    R0 = redeg.rho_curves(W, 0.0)
    out["rho_0"], out["rho_5"] = R0, redeg.rho_curves(W, 5.0)
    out["drho"], out["obs"], out["nbi"] = redeg.drho_ds0(W, redeg.ETA_PIX), redeg.observability(W), redeg.narrowband_index(W)
    for i, v in enumerate(redeg.knee_fit(R0)):
        out[f"knee_{i}"] = v
    for i, v in enumerate(redeg.knee_fit_noise(R0)):
        out[f"kneen_{i}"] = v
    for i, v in enumerate(redeg.tail_fit(spectra_from_coeffs(_noisy_coeffs(tiles, 0.0, 4.0, ("equiv_noise",))))):
        out[f"tail_{i}"] = v
    W_src = W[:N_SRC]
    cal = redeg.calibrate(W_src)
    out["cal1"], out["cal2"] = cal.cal1, cal.cal2
    targets = {"clean": W[N_SRC:] * np.exp(-6.0 * tab.lamb)[None, None, :],
               "noisy": spectra_from_coeffs(_noisy_coeffs(tiles[N_SRC:], 6.0, 6.0, ("equiv_tnoise",)))}
    for name, Wt in targets.items():
        o, ks, res = redeg.identify_spectra(Wt, W_src, cal)
        for k in ("s_hat", "p_gof", "flagged", "s_use", "sig_use", "s_v11", "sig_v11", "gamma_med", "knee_med"):
            out[f"id_{name}_{k}"] = np.asarray(float(o[k]))
        out[f"id_{name}_res"] = res
        if o["flagged"] and np.isfinite(o["s_v11"]):
            out[f"id_{name}_p2"] = np.asarray(redeg.noise_model_pvalue(tiles[:N_SRC], res, o["s_v11"], o["sig_v11"], ("equiv", name)))
    etas = np.geomspace(0.01, 1e4, 600)
    rows = []
    for alpha in (1.5, 2.0, 2.5, 3.0):
        W0 = spectra_from_coeffs(powerlaw_coeffs(alpha, 16, rng_for("powerlaw", alpha)))
        rat = []
        for s0 in (2.0, 4.0, 8.0, 16.0):
            R = redeg.rho_curves(W0, s0, etas)
            rat += [redeg.half_point(R[i], etas) / s0 for i in range(16)]
        rows.append(np.median(rat))
    out["b1_half_energy"] = np.array(rows)
    Zs, Zt, Zl, ys = inp["Zs"], inp["Zt"], inp["Zl"], inp["ys"]
    T_fit, nll1, nllT = redeg.fit_temperature(Zs, ys)
    out["temp"] = np.array([T_fit, nll1, nllT])
    T = T_fit if T is None else float(T)
    Ps, Pt, Pl = (redeg.softmax_T(Z, T) for Z in (Zs, Zt, Zl))
    out["negent"], out["nuc"] = E.negent(Pt), np.asarray(E.nuc_score(Pt))
    out["mano"] = np.array([E.mano_score(Zt / T, True), E.mano_score(Zt / T, False), E.mano_phi(Zs / T)])
    try:
        import ot  # noqa: F401
        keys = ("ATC", "ATC_NE", "DoC", "COT", "COTT")
        lv = redeg.plugins_at_level(Pt, Pl, ys, Ps, target_key=("equiv_target",), level_key=("lvl", 4.0, 0.0))
        z = redeg.plugins_at_level(Pt, Ps, ys, Ps, target_key=("equiv_target",), level_key=("lvl", 0.0, 0.0))
        out["plugins_level"] = np.array([lv[k + "@"] for k in keys])
        out["plugins_zero"] = np.array([z[k + "@"] for k in keys])
        out["baselines"] = np.array([lv[k] for k in keys])
    except ImportError:
        pass
    from PIL import Image
    rows = []
    for q in (50, 85, 95):
        for sub in (0, 1, 2):
            buf = io.BytesIO()
            Image.fromarray(tiles[0]).save(buf, format="JPEG", quality=q, subsampling=sub)
            st = jpeg_file_store(buf.getvalue())
            rows.append([q, sub, st[0], st[1], int(st[2])])
    out["store_rows"] = np.array(rows)
    out["jpeg_rt"] = _sub(jpeg_roundtrip(tiles[:4], 85, 2))
    levels = redeg.FSI_LEVELS
    Zc = {float(s): inp["fsi_Zc"][i] for i, s in enumerate(levels)}
    Zn = {float(s): inp["fsi_Zn"][i] for i, s in enumerate(levels)}
    f = redeg.floor_sensitivity_index(Zc, {s: Z.argmax(1) == ys for s, Z in Zc.items()}, Zn,
                                      {s: Z.argmax(1) == ys for s, Z in Zn.items()}, 0.9)
    out["fsi"] = np.array([f["fsi"]] + [f["per_level"][float(s)] for s in levels])
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_prereg import b27_canonical

    rec = json.loads((ROOT / "prereg" / "b27_registration_record.json").read_text(encoding="utf-8"))
    out["b27_sha256"] = np.asarray(hashlib.sha256(b27_canonical(rec).encode("utf-8")).hexdigest())
    out["v5_level"] = np.array([redeg.plugin_level(d) for d in _V5_CASES])
    return out


def compare(ref: dict, pkg: dict, rtol: float = 0.0, atol: float = 0.0, verbose: bool = True) -> list:
    """Keys whose values differ (exact comparison by default; NaN equals NaN)."""
    bad = []
    for k in sorted(ref):
        a = np.asarray(ref[k])
        if k not in pkg:
            bad.append(k)
            if verbose:
                print(f"  {k:22s} missing in the package output")
            continue
        b = np.asarray(pkg[k])
        if a.dtype.kind in "USO" or b.dtype.kind in "USO":
            same, diff = bool(np.all(a == b)), 0.0
        elif a.shape != b.shape:
            same, diff = False, np.inf
        elif a.dtype.kind in "biu" and b.dtype.kind in "biu":
            same, diff = bool(np.array_equal(a, b)), float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        else:
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            same = bool(np.allclose(a64, b64, rtol=rtol, atol=atol, equal_nan=True)) if (rtol or atol) else \
                bool(np.array_equal(a64, b64, equal_nan=True))
            fin = np.isfinite(a64) & np.isfinite(b64)
            diff = float(np.max(np.abs(a64[fin] - b64[fin]))) if fin.any() else 0.0
        if not same:
            bad.append(k)
        if verbose:
            print(f"  {k:22s} {str(a.shape):14s} {'identical' if same and diff == 0 else ('ok' if same else 'DIFFERENT')}"
                  + (f"  (max |diff| {diff:.1e})" if diff else ""))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the redeg package against the final notebook.")
    ap.add_argument("--notebook", type=Path, help="the final notebook (.ipynb)")
    ap.add_argument("--fixture", type=Path, help="compare against a stored fixture instead of a notebook")
    ap.add_argument("--write-fixture", type=Path, help="write the notebook's outputs to this .npz file")
    args = ap.parse_args()
    inp = make_inputs()
    if args.notebook:
        ref = notebook_outputs(build_namespace(args.notebook), inp)
        if args.write_fixture:
            args.write_fixture.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.write_fixture, **{k: np.asarray(v) for k, v in ref.items()})
            print(f"fixture written: {args.write_fixture} ({args.write_fixture.stat().st_size / 1e3:.0f} kB)")
    elif args.fixture:
        with np.load(args.fixture, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
    else:
        ap.error("give --notebook or --fixture")
    print("notebook vs package:")
    bad = compare(ref, package_outputs(inp))
    print(f"\nhalf-energy law, block B1 (eta_1/2 / s0 for alpha = 1.5, 2, 2.5, 3): {np.round(ref['b1_half_energy'], 3).tolist()}"
          "   published: [0.72, 0.961, 1.489, 3.416]")
    print("ALL IDENTICAL" if not bad else f"{len(bad)} quantities differ: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
