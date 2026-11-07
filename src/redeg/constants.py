"""Registered constants.

Every value below was fixed before the run that first used it, either in a registration document (``prereg/``) or in the
notebook code that a document adopts by reference; the comments say which. Changing one of them changes the method that the
paper evaluates.
"""
from __future__ import annotations

import numpy as np

# ---- frozen method v1.0 (prereg/v1.0.json, "frozen_method_v1_0") ----
TILE = 224                                                   # tiles are 224 x 224 RGB, in digital numbers (DN, 0-255)
N_LAMBDA_BINS = 3000                                         # log-spaced bins of the Laplacian eigenvalues
ETA_PIX = np.geomspace(0.25, 1024.0, 13)                     # re-blur levels eta (px^2)
S_FIT = np.geomspace(0.05, 2048, 170)                        # knee fit: grid of s (px^2)
G_FIT = np.linspace(0.3, 2.2, 39)                            # knee fit: grid of gamma
R_FIT = np.concatenate([[0.0], np.geomspace(1e-3, 30, 15)])  # v1.0 joint blur + noise fit: noise-to-signal ratios
CAL_GRID = np.concatenate([[0.0], np.geomspace(0.05, 400, 30)])  # calibration levels of the labelled source (px^2)
GOF_ALPHA = 0.01                                             # fit check: one-sided Mann-Whitney U, level 0.01
HIGH_BAND_LAMBDA_MIN = 6.0                                   # noise is read from the DCT band lambda >= 6
NOISE_DN = 3.0                                               # sensor-noise level of the in-family suite (DN)
N_EPS = 1e-6                                                 # epsilon of the pipeline standardisation
S_ACC = np.array([0., 1., 2., 4., 8., 16., 32., 64., 128., 256.])  # accuracy-curve grid (px^2)

# ---- addendum v2 (prereg/v2.json, "v2_path"): the noise-aware path ----
# v2 adopts the v1.1 tail fit, a reported variant already in the notebook of run 1 (Cell 3.2), with that code's grids.
S2 = np.geomspace(0.05, 2048, 200)                           # tail fit (v1.1): grid of s
G2 = np.linspace(0.3, 2.5, 45)                               # tail fit (v1.1): grid of gamma
NOISE_MODEL_MAX_REF = 480                                    # noise-model check (v2 code, Cell 3.2): at most 480 source images

# ---- addendum v6 (prereg/v6.json): temperature scaling, MaNo, attribution ----
TEMPERATURE_RANGE = (0.05, 20.0)                             # bounded NLL search over log T
MANO_P = 4                                                   # MaNo norm
MANO_ETA = 5.0                                               # MaNo: softmax if Phi > eta, else Taylor normalisation
PHI_MIN_GAP = 0.005                                          # phi is undefined when |c_src(0) - c_tgt| < 0.005
PHI_SWITCH = 0.5                                             # the switch uses X@s_hat when phi >= 0.5

# ---- addendum v7 (prereg/v7.json, "floor_sensitivity") ----
FSI_LEVELS = (1.0, 4.0, 16.0)                                # blur levels of the floor-sensitivity index (px^2)
FSI_THRESHOLD_PP = 4.0                                       # FSI >= 4 pp predicts harm from a floating-point re-degradation
