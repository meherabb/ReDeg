#!/usr/bin/env python3
"""Recompute the SHA-256 of every registration document and of the B27 registration record.

The hash of a document is the SHA-256 of its canonical JSON serialisation,

    json.dumps(doc, sort_keys=True, separators=(",", ":"))      (UTF-8, ASCII escapes)

which is exactly how the notebook computed it before each run (Cell 1.4). The pretty-printed files in
``prereg/`` therefore hash to the registered values although their bytes differ from the canonical form.

Each hash is checked against three independent copies:
  1. the values printed in the paper (Table S34 and Supplement G), hard-coded below;
  2. ``prereg/registered_sha256.json``;
  3. the run manifest written by the final run (``results/run_manifest.json``), which embeds every document.

Standard library only.  Usage:  python scripts/verify_prereg.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Table S34 of the paper (and the notebook's introduction): document -> (frozen, registered SHA-256).
PAPER = {
    "v1.0": ("before run 1", "58594e8ccc059e4827b22061855cf0a139958113e5e114cce963d1b8b6aebfb5"),
    "v2": ("after run 1, before run 2", "9e3ab0852200bfaabed8b711ad71453836e2649c981f3f92b84836781c863998"),
    "v3": ("after run 2, before run 3", "86ab550b0b1a2024c8dbfee00a6a7cd1014aca3c0993acc13fc9af1b07420faf"),
    "v4": ("after run 3, before run 4", "b885f994c5d840ee714668860f7e21f12ab71f673ef360be6319e2a72639d209"),
    "v5": ("after run 4, before run 5", "ecec36131e8b3e7755ece91f6548e338b88522c163c9eb7f8d481011de57fc3c"),
    "v6": ("after run 5, before run 6", "b66b2997d9861bb358d2beb4de7f3ed3e64e2cc7ad214f2a7e31c2ddc97d6580"),
    "v7": ("after run 7, before run 8", "be1de05049d2b835c2e538ed57429bd3a4cd4da4d34c560908a4d53f8af9b131"),
    "v8": ("after run 8, before run 9", "9dbd90fa80627c4f47f9a9076d6c2c67e178cbb97912f834a29f1260bb027cb2"),
}
PAPER_B27 = "e7c59048e412a5a376e40d5fbc241323e264ba02abfd1979bc09c583bd57078b"
MANIFEST_KEYS = {"v1.0": "prereg", "v2": "addendum", "v3": "addendum3", "v4": "addendum4", "v5": "addendum5",
                 "v6": "addendum6", "v7": "addendum7", "v8": "addendum8"}


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def b27_canonical(record: dict) -> str:
    """The canonical string of the B27 record (Cell 6.4m, ``_canonical_registration``): FSI rounded to 6 decimals and the prediction."""
    models = record["models"]
    return json.dumps({m: dict(fsi=round(float(r["fsi"]), 6), prediction=r["prediction"]) for m, r in sorted(models.items())},
                      sort_keys=True, separators=(",", ":"))


def _mark(c) -> str:
    return "  -  " if c is None else ("  ok " if c else " FAIL")


def main() -> int:
    registered = json.loads((ROOT / "prereg" / "registered_sha256.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "results" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    ok = True
    print(f"{'document':10s} {'frozen':27s} {'SHA-256 (recomputed)':66s} paper  list  manifest")
    for ver, (frozen, h_paper) in PAPER.items():
        doc = json.loads((ROOT / "prereg" / f"{ver}.json").read_text(encoding="utf-8"))
        h = canonical_sha256(doc)
        c1 = h == h_paper
        c2 = h == registered["documents"].get(ver)
        if manifest is not None:
            key = MANIFEST_KEYS[ver]
            c3 = h == manifest.get(f"{key}_sha256") and canonical_sha256(manifest[key]) == h
        else:
            c3 = None
        ok &= c1 and c2 and (c3 is not False)
        print(f"{ver:10s} {frozen:27s} {h}  {_mark(c1)} {_mark(c2)} {_mark(c3)}")

    rec = json.loads((ROOT / "prereg" / "b27_registration_record.json").read_text(encoding="utf-8"))
    hb = hashlib.sha256(b27_canonical(rec).encode("utf-8")).hexdigest()
    b_ok = hb == PAPER_B27 == rec.get("sha256") == registered.get("b27_registration_record")
    if manifest is not None:
        b_ok &= hb == manifest.get("v8_registration", {}).get("sha256")
    ok &= b_ok
    print(f"\nB27 registration record (created {rec.get('created')}, before any target image was read: {rec.get('before_targets')})")
    print(f"  SHA-256 recomputed from the {len(rec['models'])} registered indices: {hb}  {'ok' if b_ok else 'FAIL'}")
    thr = float(rec.get("threshold", float('nan')))
    for m, r in sorted(rec["models"].items(), key=lambda kv: -kv[1]["fsi"]):
        print(f"    {m:45s} FSI {r['fsi']:+7.2f} pp  ->  {r['prediction']:8s} (threshold {thr:g} pp)")
    print("\nALL HASHES MATCH" if ok else "\nMISMATCH: see the rows marked FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
