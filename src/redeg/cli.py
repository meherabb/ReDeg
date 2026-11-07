"""Command line: ``redeg identify --source DIR --target DIR`` and ``redeg storage DIR``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .identify import BlurIdentifier
from .io import list_images, load_tiles
from .storage import read_store, store_label


def _identify(args) -> int:
    src_paths, tgt_paths = list_images(args.source), list_images(args.target)
    if not src_paths or not tgt_paths:
        print("no images found in the source or the target folder", file=sys.stderr)
        return 2
    t0 = time.time()
    ident = BlurIdentifier(joint_fit=not args.no_joint).fit(load_tiles(src_paths, resample=args.resample))
    t1 = time.time()
    tgt = load_tiles(tgt_paths, resample=args.resample)
    res = ident.identify(tgt, key=(args.key,))
    t2 = time.time()
    store = read_store([str(p) for p in tgt_paths], cap=args.store_cap)
    out = dict(res.as_dict(), store=store, store_label=store_label(store), n_source=len(ident.src_tiles),
               seconds_calibration=round(t1 - t0, 2), seconds_identification=round(t2 - t1, 2))
    if args.json:
        print(json.dumps(out, indent=2, default=float))
        return 0
    print(f"source images        {out['n_source']}   (calibration {out['seconds_calibration']} s)")
    print(f"target images        {res.n}   (identification {out['seconds_identification']} s)")
    print(f"blur level s_hat     {res.s_hat:.3f} px^2   (PSF sd {res.s_hat ** 0.5:.3f} px)")
    verdict = "FLAGGED: blur alone does not explain the target" if res.flagged else "blur model accepted"
    print(f"fit check            p = {res.p_fit:.3g}   -> {verdict}")
    if res.flagged:
        print(f"noise-aware path     s = {res.s_tail:.3f} px^2, sigma = {res.sigma_tail:.2f} DN, "
              f"noise-model check p = {res.p_noise:.3g}   ({'rejected' if res.noise_rejected else 'accepted'})")
    print(f"re-degrade source to s5 = {res.s5:.3f} px^2, sigma5 = {res.sigma5:.2f} DN, stored as {out['store_label']}")
    return 0


def _storage(args) -> int:
    paths = [str(p) for p in list_images(args.folder)]
    if not paths:
        print("no images found", file=sys.stderr)
        return 2
    st = read_store(paths, cap=args.cap)
    print(json.dumps(dict(st, label=store_label(st)), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="redeg", description="Re-degradation: label-free blur identification in px^2.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("identify", help="identify the blur of a target folder relative to a labelled source folder")
    pi.add_argument("--source", required=True, help="folder of source images (labels are not needed)")
    pi.add_argument("--target", required=True, help="folder of unlabelled target images")
    pi.add_argument("--key", default="target", help="name of the random streams of the noise-model check")
    pi.add_argument("--store-cap", type=int, default=None, help="read the storage from at most this many target files")
    pi.add_argument("--no-joint", action="store_true", help="skip the (reported, superseded) v1.0 joint blur+noise fit")
    pi.add_argument("--resample", choices=["bicubic", "bilinear"], default="bicubic",
                    help="interpolation of the resize to 256 px (bicubic: scene datasets; bilinear: the ImageNet blocks)")
    pi.add_argument("--json", action="store_true", help="print the result as JSON")
    pi.set_defaults(func=_identify)
    ps = sub.add_parser("storage", help="read the storage step (8-bit, JPEG quality and chroma subsampling) of a folder")
    ps.add_argument("folder")
    ps.add_argument("--cap", type=int, default=None)
    ps.set_defaults(func=_storage)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
