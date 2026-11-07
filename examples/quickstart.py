"""Quick start: identify the blur of unlabelled images in px^2 and re-degrade the labelled source like the target.

Runs in a few seconds on synthetic natural-image-like tiles (isotropic 1/f^2 power spectra), so nothing is downloaded:

    python examples/quickstart.py

For your own images (any size; resized to 256 px on the short side and centre-cropped to 224, as in the experiments) use
the command line: ``redeg identify --source SOURCE_DIR --target TARGET_DIR``.
"""
import io

import numpy as np
from PIL import Image

from redeg import BlurIdentifier, read_store, redegrade, store_label
from redeg._keys import rng_for
from redeg.semigroup import idct2, tables_for
from redeg.storage import jpeg_bytes
from redeg.theory import powerlaw_coeffs


def natural_tiles(n: int, key: str) -> np.ndarray:
    """uint8 (n, 224, 224, 3) tiles with |omega|^-2 power spectra, the second-order statistics of natural images."""
    c = powerlaw_coeffs(2.0, n, rng_for(key), channels=3) * np.exp(-0.5 * 0.7 * tables_for(224).lam)
    return np.clip(np.round(idct2(c) * 25.0 + 120.0), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


source = natural_tiles(48, "source")              # the labelled source (its labels are never needed here)
scene = natural_tiles(48, "scene")                # what another camera looks at

# three ways the target could have been formed; the true blur is a Gaussian PSF of variance 6 px^2 (sd 2.45 px)
float_target = redegrade(scene, 6.0)                                              # T_6 x, floating point
noisy_8bit = redegrade(scene, 6.0, 2.0, dict(kind="lossless8"), key=("camera",))  # Q8(T_6 x + 2 xi)
jpeg_files = [jpeg_bytes(t) for t in redegrade(scene, 6.0, 0.0, dict(kind="lossless8"))]  # JPEG 85 files
jpeg_target = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpeg_files])

ident = BlurIdentifier().fit(source)              # calibration on the source pixels: 31 levels in [0, 400] px^2
print(f"{'target formation':38s} {'s_hat':>6s} {'flagged':>8s} {'s5':>6s} {'sigma5':>7s}  storage read from the target")
for name, target, store in (("blur 6 px^2, floating point", float_target, dict(kind="float")),
                            ("blur 6 px^2 + 2 DN noise, 8-bit", noisy_8bit, read_store(noisy_8bit, "lossless8")),
                            ("blur 6 px^2, 8-bit, JPEG files", jpeg_target, read_store(jpeg_files))):
    res = ident.identify(target)                  # pixels only: the same answer for every classifier
    print(f"{name:38s} {res.s_hat:6.2f} {('yes' if res.flagged else 'no'):>8s} {res.s5:6.2f} {res.sigma5:7.2f}  {store_label(store)}")

# the labelled source, re-degraded to the identified level and stored like the target: D(x) = JPEG_85(Q8(T_s5 x))
source_like_target = redegrade(source, res.s5, res.sigma5, read_store(jpeg_files))
print(f"\nre-degraded source: {source_like_target.shape} {source_like_target.dtype}; calibrate any label-free accuracy "
      "estimator on it (redeg.plugins_at_level) or compare confidences (redeg.attribution_score).")
