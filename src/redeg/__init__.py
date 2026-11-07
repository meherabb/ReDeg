"""redeg: re-degradation for label-free identification and attribution of blur shift.

Reference implementation of the method of *Re-Degradation Reveals Resolution: Label-Free Identification and Attribution
of Blur Shift* (anonymous CVPR 2027 submission), extracted from the notebook that produced every number of the paper
(see ``notebooks/``) with the registered constants (see ``prereg/``).

Quick start::

    from redeg import BlurIdentifier, read_store, redegrade

    ident = BlurIdentifier().fit(source_tiles)          # uint8 (n, 224, 224, 3); no labels needed
    res = ident.identify(target_tiles)                  # res.s_hat (px^2), res.flagged, res.s5, res.sigma5
    store = read_store(target_files)                    # {"kind": "jpeg", "q": 85, "sub": 2, ...}
    source_like_target = redegrade(source_tiles, res.s5, res.sigma5, store)
"""
from .attribution import attribution_score, blend, labelled_share, switch
from .calibration import fit_temperature, softmax, softmax_T
from .constants import CAL_GRID, ETA_PIX, FSI_LEVELS, FSI_THRESHOLD_PP, GOF_ALPHA
from .fsi import floor_sensitivity_index, fsi_inputs, predicts_harm
from .identify import BlurIdentifier, Calibration, Identification, calibrate, identify_spectra, noise_model_pvalue, plugin_level
from .knee import knee_fit, knee_fit_noise, tail_fit
from .plugins import TargetCosts, bias_transfer, plugins_at_level
from .response import drho_ds0, half_point, narrowband_index, observability, rho_curves
from .semigroup import PixelTables, dct2, heat, idct2, spectra, standardize, tables_for
from .storage import jpeg_file_store, jpeg_roundtrip, quantize8, read_store, redegrade, store_label

__version__ = "1.0.0"

__all__ = [
    "BlurIdentifier", "Identification", "Calibration", "calibrate", "identify_spectra", "noise_model_pvalue", "plugin_level",
    "heat", "dct2", "idct2", "standardize", "spectra", "tables_for", "PixelTables",
    "rho_curves", "drho_ds0", "observability", "narrowband_index", "half_point",
    "knee_fit", "knee_fit_noise", "tail_fit",
    "read_store", "jpeg_file_store", "jpeg_roundtrip", "quantize8", "redegrade", "store_label",
    "softmax", "softmax_T", "fit_temperature",
    "attribution_score", "labelled_share", "switch", "blend",
    "plugins_at_level", "TargetCosts", "bias_transfer",
    "floor_sensitivity_index", "fsi_inputs", "predicts_harm",
    "ETA_PIX", "CAL_GRID", "GOF_ALPHA", "FSI_LEVELS", "FSI_THRESHOLD_PP",
]
