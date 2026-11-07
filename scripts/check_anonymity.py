#!/usr/bin/env python3
# anonymity-scan: skip-file  (this file contains the patterns it searches for)
"""Scan files for anything that could identify the authors before the repository is made public.

Searches text files, notebook sources and text outputs (not embedded images), and PDF metadata for e-mail addresses, local
paths (home folders, mounted drives, Kaggle input slugs), access tokens, Kaggle/Papermill/Colab notebook metadata and,
optionally, your own names and handles listed one per line in a file that you do not commit (``.anonymity_names.txt`` is
git-ignored):

    python scripts/check_anonymity.py                                 # the whole repository
    python scripts/check_anonymity.py notebooks --names-file .anonymity_names.txt

Exit code 1 when something is found. Standard library only; nothing is executed or modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "build", "dist", ".venv", "venv"}
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".npz", ".npy", ".pkl", ".zip", ".gz", ".tar", ".whl", ".ico", ".pt", ".pth"}
TLD = r"(?:com|org|net|edu|gov|io|ai|ac\.[a-z]{2}|edu\.[a-z]{2}|co\.[a-z]{2}|[a-z]{2})"
PATTERNS = {
    "e-mail address": re.compile(r"(?<![\w@])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\." + TLD + r"\b"),
    "home folder": re.compile(r"(?:/home/(?!user\b|runner\b|jovyan\b)[A-Za-z][\w.-]*|/Users/[A-Za-z][\w.-]*"
                              r"|[A-Z]:\\\\?Users\\\\?[^\\\s\"']+)"),
    "mounted drive": re.compile(r"/content/drive|/MyDrive|/gdrive"),
    "Kaggle input slug": re.compile(r"/kaggle/input/[A-Za-z0-9][\w.-]*"),
    "Kaggle profile": re.compile(r"kaggle\.com/(?!datasets/|competitions/|code/|docs|models/)[A-Za-z0-9_-]{3,}"),
    "access token": re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
                               r"|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{32,})\b"),
    "credential file": re.compile(r"kaggle\.json|KAGGLE_KEY|HF_TOKEN\s*="),
}
NB_META_KEYS = ("kaggle", "papermill", "colab")


def load_names(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def scan_text(text: str, names: list[str]):
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            yield label, m.group(0)
    low = text.lower()
    for n in names:
        i = low.find(n.lower())
        if i >= 0:
            yield "your name / handle", text[i:i + len(n)]


def notebook_text(nb: dict) -> tuple[str, list[str]]:
    """Sources and text outputs of a notebook (images and HTML-rendered tables excluded) and its metadata findings."""
    parts, meta = [], []
    for k in NB_META_KEYS:
        if k in nb.get("metadata", {}):
            meta.append(f"notebook metadata '{k}'")
    for i, c in enumerate(nb.get("cells", [])):
        parts.append("".join(c.get("source", "")))
        for k in NB_META_KEYS:
            if k in (c.get("metadata") or {}):
                meta.append(f"cell {i} metadata '{k}'")
        for o in c.get("outputs", []) or []:
            if "text" in o:
                parts.append("".join(o["text"]) if isinstance(o["text"], list) else str(o["text"]))
            for mime, v in (o.get("data") or {}).items():
                if mime.startswith("text/plain"):
                    parts.append("".join(v) if isinstance(v, list) else str(v))
            if o.get("output_type") == "error":
                parts.append(" ".join(o.get("traceback", [])))
    return "\n".join(parts), sorted(set(meta), key=meta.index)


def pdf_metadata(raw: bytes) -> str:
    """The values of the document information entries of a PDF (Author, Creator, Producer, Title, Subject, Keywords)."""
    out = []
    for key in (b"Author", b"Creator", b"Producer", b"Title", b"Subject", b"Keywords"):
        for m in re.finditer(rb"/" + key + rb"\s*\(([^)]{0,200})\)", raw):
            out.append(f"{key.decode()}: {m.group(1).decode('latin-1')}")
    return "\n".join(out)


def scan_file(path: Path, names: list[str]) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in BINARY:
        return []
    raw = path.read_bytes()
    if suffix == ".pdf":
        meta = pdf_metadata(raw)
        return [f"PDF metadata -> {label}: {hit}" for label, hit in scan_text(meta, names)] + \
               [f"PDF author field: {ln}" for ln in meta.splitlines() if ln.startswith("Author:") and ln.strip() != "Author:"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    if "anonymity-scan: skip-file" in text[:400]:
        return []
    found = []
    if suffix == ".ipynb":
        try:
            text, meta = notebook_text(json.loads(text))
        except json.JSONDecodeError:
            meta = ["not valid JSON"]
        found += meta
    found += [f"{label}: {hit}" for label, hit in scan_text(text, names)]
    return found


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.relative_to(p).parts) & SKIP_DIRS) and not any(x.endswith(".egg-info") for x in f.parts):
                yield f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Search files for author-identifying content.")
    ap.add_argument("paths", nargs="*", default=[str(ROOT)])
    ap.add_argument("--names-file", type=Path, default=ROOT / ".anonymity_names.txt",
                    help="file with your names, user names, e-mail and affiliation, one per line (never commit it)")
    args = ap.parse_args(argv)
    names = load_names(args.names_file)
    n_files, bad = 0, 0
    for f in iter_files(args.paths):
        n_files += 1
        hits = scan_file(f, names)
        if hits:
            bad += 1
            print(f"{f}")
            for h in dict.fromkeys(hits):
                print(f"    {h}")
    print(f"\n{n_files} files scanned" + (f", {len(names)} personal strings searched" if names else ", no personal strings given")
          + (f"; {bad} file(s) need attention" if bad else "; nothing found"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
