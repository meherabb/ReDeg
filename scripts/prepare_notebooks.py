#!/usr/bin/env python3
"""Name, clean and check the executed notebooks before they are added to ``notebooks/``.

Each notebook is identified from its content, not its file name: the highest registration document whose hash it embeds
(v1.0 ... v8) and the units it reports as "computed in this session". It is written as
``notebooks/EXPk/EXPk_vX[...].ipynb`` (the names of ``notebooks/README.md``; run k of the paper is EXPk, and the two
notebooks of run 4 are EXP4a and EXP4b) with every result kept. Two things are removed: the ``kaggle``, ``papermill`` and
``colab`` metadata of the notebook and of every cell (Kaggle stores dataset ids and timestamps there), and embedded copies of
files, i.e. HTML download links that carry a whole ``outputs.zip`` as a data URI (the v2 and v3 notebooks print one in their
last cell; the zip itself sits next to the notebook). The result is then scanned with ``check_anonymity.py``. Notebooks are
read as JSON; nothing is executed.

    python scripts/prepare_notebooks.py ~/Downloads/*.ipynb --out notebooks [--names-file .anonymity_names.txt] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_anonymity import NB_META_KEYS, ROOT, load_names, scan_file  # noqa: E402

HASH_PREFIX = {"v1.0": "58594e8ccc059e48", "v2": "9e3ab0852200bfaa", "v3": "86ab550b0b1a2024", "v4": "b885f994c5d840ee",
               "v5": "ecec36131e8b3e77", "v6": "b66b2997d9861bb3", "v7": "be1de05049d2b835", "v8": "9dbd90fa80627c4f"}
LEVELS = list(HASH_PREFIX)
EMBEDDED_FILE = re.compile(r"data:application/[\w.+-]+;base64,")
EMBEDDED_MIN_CHARS = 100_000


def text_outputs(nb: dict) -> str:
    out = []
    for c in nb.get("cells", []):
        for o in c.get("outputs", []) or []:
            t = o.get("text", (o.get("data") or {}).get("text/plain"))
            if t:
                out.append("".join(t) if isinstance(t, list) else str(t))
    return "\n".join(out)


def session_start(nb: dict) -> str | None:
    """Earliest cell execution time recorded by Jupyter / Kaggle / Papermill, if any (UTC, ISO)."""
    ts = []
    for c in nb.get("cells", []):
        m = c.get("metadata") or {}
        for v in list((m.get("execution") or {}).values()) + [(m.get("papermill") or {}).get("start_time")]:
            if isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT", v):
                ts.append(v[:19])
    return min(ts) if ts else None


def identify_run(nb: dict) -> tuple[str, str, set]:
    """(canonical file stem, registration level, units computed in the session) of one executed notebook."""
    src = "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []) if c.get("cell_type") == "code")
    level = next((v for v in reversed(LEVELS) if HASH_PREFIX[v] in src), None)
    if level is None:
        raise ValueError("no registration hash found: not a notebook of this project")
    lines = re.findall(r"computed in this session: ([^\n]*)", text_outputs(nb))
    units = set(re.findall(r"\b(B\d+[A-Za-z0-9_]*) done", lines[-1])) if lines else set()
    has = lambda *pre: any(u.startswith(p) for u in units for p in pre)  # noqa: E731
    if level in ("v1.0", "v2", "v3", "v5"):
        return {"v1.0": "EXP1_v1.0", "v2": "EXP2_v2", "v3": "EXP3_v3", "v5": "EXP5_v5"}[level], level, units
    if level == "v4":
        return ("EXP4b_v4" if has("B12_") else "EXP4a_v4_superseded"), level, units
    if level == "v6":
        if has("B18_"):
            return "EXP7_v6", level, units
        if has("B19_", "B20_", "B21_", "B22_", "B23_"):
            return "EXP6_v6", level, units
        start = session_start(nb) or ""
        stem = "EXP6_v6" if start.startswith("2026-10-17") else "EXP7_v6" if start.startswith("2026-10-18") else "EXP6or7_v6_unidentified"
        return stem, level, units
    if level == "v7":
        return ("EXP8_v7" if has("B24_", "B25_") else "EXP8_v7_first_start"), level, units
    return ("EXP9_v8_final" if has("B27_") else "EXP9_v8_interrupted"), level, units


def is_embedded_file(output: dict) -> bool:
    """True for an HTML output that carries a whole file as a data URI (a download link with the run's outputs.zip)."""
    html = (output.get("data") or {}).get("text/html")
    html = "".join(html) if isinstance(html, list) else (html or "")
    return len(html) >= EMBEDDED_MIN_CHARS and bool(EMBEDDED_FILE.search(html))


def clean(nb: dict) -> dict:
    for k in NB_META_KEYS + ("widgets",):
        nb.get("metadata", {}).pop(k, None)
    for c in nb.get("cells", []):
        for k in NB_META_KEYS:
            (c.get("metadata") or {}).pop(k, None)
        if c.get("outputs"):
            c["outputs"] = [o for o in c["outputs"] if not is_embedded_file(o)]
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Identify, clean and check executed notebooks.")
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks")
    ap.add_argument("--names-file", type=Path, default=ROOT / ".anonymity_names.txt")
    ap.add_argument("--dry-run", action="store_true", help="identify and report only; write nothing")
    args = ap.parse_args(argv)
    names = load_names(args.names_file)
    plan = {}
    for p in args.notebooks:
        nb = json.loads(p.read_text(encoding="utf-8"))
        stem, level, units = identify_run(nb)
        if stem in plan:
            if plan[stem][1] == nb:
                print(f"note: {p.name} is an identical copy of {plan[stem][0].name}; skipped")
                continue
            print(f"error: {p.name} and {plan[stem][0].name} both identify as {stem}; pass one of them", file=sys.stderr)
            return 2
        plan[stem] = (p, nb, level, units)
    rc = 0
    print(f"{'input':44s} {'-> notebooks/':38s} {'reg.':5s} {'embedded':>8s}  units computed")
    for stem in sorted(plan, key=lambda s: (int(re.match(r"EXP(\d)", s).group(1)), s)):
        p, nb, level, units = plan[stem]
        n_embedded = sum(is_embedded_file(o) for c in nb.get("cells", []) for o in c.get("outputs", []) or [])
        folder = stem.split("_")[0]
        print(f"{p.name[:44]:44s} {folder + '/' + stem + '.ipynb':38s} {level:5s} {n_embedded:8d}  {', '.join(sorted(units)) or '-'}")
        if args.dry_run:
            continue
        dst = args.out / folder / f"{stem}.ipynb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(clean(nb), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        hits = scan_file(dst, names)
        for h in dict.fromkeys(hits):
            print(f"    CHECK: {h}")
        rc |= bool(hits)
    if not args.dry_run:
        print("\nwritten to", args.out, "| nothing to check" if rc == 0 else "| review the lines marked CHECK before publishing")
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
