# Method reference

The method as registered, step by step, with the constants of the registration documents and the place where each step lives
in the package and in the final notebook. Equation and theorem numbers refer to the paper.

## 1. Exact re-degradation (Eq. 1)

An image channel `x` of size H x W (224 x 224 in the experiments) is blurred by the heat semigroup
`T_s = exp(−(s/2) L)`, where `L = DᵀD` is the Laplacian of the pixel grid with reflecting boundaries and `D` takes forward
differences. The orthonormal 2-D DCT-II diagonalises `L` with eigenvalues

    λ_{k,l} = 4 sin²(πk / 2H) + 4 sin²(πl / 2W)  ∈ [0, 8),

so `T_s` multiplies each DCT coefficient by `exp(−sλ/2)`. `T_s` is exact on the finite tile, `T_s T_η = T_{s+η}`, and `s` is
the per-axis variance (px²) of the Gaussian-equivalent PSF; a Gaussian PSF with standard deviation `σ_px` has `s = σ_px²`.

Package: `redeg.heat`, `redeg.semigroup.PixelTables` · Notebook: Cell 2.1 (`heat`, `dct2`), Cell 6.4c (`pixel_tables(N)` for
32-px images).

## 2. Response (Eq. 2, Lemma 1, Theorem 2)

With `c = DCT(x)` and `ν_x = Σ_{λ>0} λ c_λ² δ_λ` the gradient-power measure, the gradient energy after a re-blur by `η` is
`e_x(η) = ‖D T_η x‖² = Σ λ c_λ² e^{−ηλ}`, and the response is `ρ_x(η) = 1 − e_x(η)/e_x(0)`, averaged over the colour channels.
The eigenvalues are grouped into 3000 log-spaced bins (`W`, shape `(n, 3, 3000)`), which makes `ρ` at any level a matrix
product. Per-image gain and offset do not change `ρ` (Lemma 1).

Package: `redeg.spectra`, `redeg.rho_curves`, `redeg.drho_ds0` (sensitivity identity, Theorem 2b),
`redeg.observability` (`O_x`), `redeg.narrowband_index` · Notebook: Cell 2.1.

## 3. Knee fit

Every image gets `ρ` at 13 re-blur levels `η ∈ geomspace(0.25, 1024, 13)` px² and a grid least-squares fit of the knee law

    ρ(η) = 1 − (1 + η/s)^(−γ),     s ∈ geomspace(0.05, 2048, 170),   γ ∈ linspace(0.3, 2.2, 39),

which returns the raw knee `s_raw`, the exponent and the residual sum of squares (SSE). For a power-law gradient spectrum the
knee law holds up to the incomplete-gamma bound of Theorem 2(e), and for natural images (`γ ≈ 1`) a re-blur by the existing
blur removes half of the remaining gradient energy (Corollary 1).

Package: `redeg.knee_fit` · Notebook: Cell 3.1 (`knee_fit`, asserted equal to the v1.0 reference loop `knee_fit_ref`).

## 4. Calibration and identification (frozen v1.0)

The knee reflects both the blur and the smoothness of the scene. Calibration removes the scene part: the labelled source is
blurred to each of 31 levels `CAL_GRID = {0} ∪ geomspace(0.05, 400, 30)` px², and `CAL(s)` is its median knee. Then

    ŝ = CAL⁻¹(median target knee),

inverted by linear interpolation on the running maximum of `CAL`. Only pixels are used, so `ŝ` is the same for every
classifier.

**Fit check.** A one-sided Mann-Whitney test compares the target's knee-fit SSE with the SSE of the source blurred to `ŝ`; the
condition is *flagged* when the target fits worse (`p < 0.01`). It is a heuristic check, not an exact test (the paper reports how
it depends on the number of images, Table S7).

Package: `redeg.calibrate`, `redeg.identify_spectra`, `redeg.BlurIdentifier` · Notebook: Cell 3.2 (`build_cal`, `inv`,
`identify`).

## 5. Noise-aware path (addendum v2)

For flagged conditions only:

1. **Tail fit (v1.1).** The noise variance `σ²` is the median over the bins with `λ ≥ 6` of `W / (count · λ)` (channel mean).
   Its energy `C σ² Σ_b count_b λ_b e^{−ηλ_b}` (C channels) is subtracted from the observed energy at every level; levels where
   the noise exceeds 10% of the observed energy are dropped, and at least four must remain. `log E(η) = a − γ log(s + η)` is
   fitted on `s ∈ geomspace(0.05, 2048, 200)`, `γ ∈ linspace(0.3, 2.5, 45)`, and calibrated on the clean source restricted
   to the same range of levels. This gives `(ŝ, σ̂)`.
2. **Noise-model check.** The SSE of the target is compared (one-sided Mann-Whitney, 0.01) with the SSE of at most 480 source
   images simulated at `(ŝ, σ̂)` with fresh Gaussian noise, added in the DCT domain (exact, since the orthonormal DCT of white
   noise is white). The noise level is kept only if this check accepts the model.

The v1.0 joint blur + noise fit (`knee_fit_noise`, noise-to-signal ratios `{0} ∪ geomspace(1e-3, 30, 15)`) is registered and
reported but superseded by this path.

**Level used to re-degrade the source (addendum v5).** `s5` = the tail-fit level when flagged, else `ŝ`; `σ5` = the
tail-fit noise when flagged *and* the noise model is accepted, else 0.

Package: `redeg.tail_fit`, `redeg.identify.identify_tail`, `redeg.noise_model_pvalue`, `redeg.plugin_level`;
`Identification.s5`, `.sigma5` · Notebook: Cell 3.2 (`tail_fit`, `identify_v11`, `noise_model_pvalue`), Cell 6.4k (`v5_level`).

## 6. Storage-matched re-degradation (addendum v7)

A target that was blurred and then stored is not `T_s x`. The labelled source is re-degraded as

    D(x) = decode(JPEG_{q,c}(Q8(T_{s5} x + σ5 ξ))),

where `Q8` rounds to 8 bits and clips to [0, 255] and `ξ` is fresh white noise. The JPEG quality `q` is the one whose IJG
quantisation tables (as written by Pillow) equal the file's tables, or the nearest luminance table in L1; the chroma
subsampling `c` comes from the sampling factors; over a target set the modal pair is used. The JPEG step is dropped for
lossless 8-bit targets and `Q8` for floating-point ones. For floating-point targets, and for lossless targets at
`s5 = σ5 = 0`, the storage-matched source is the floating-point one.

Package: `redeg.read_store`, `redeg.storage.jpeg_file_store`, `redeg.redegrade` · Notebook: Cell 5.11 (`jpeg_file_store`,
`read_store`, `jpeg_roundtrip7`), Cell 6.4k (`storage_images`, `_st_exact`).

## 7. Temperature scaling (addendum v6)

One temperature per classifier minimises the mean negative log-likelihood of the clean labelled source over
`log T ∈ [log 0.05, log 20]` (bounded scalar search, tolerance 1e-4). Every softmax of every estimator, plug-in and score uses
`Z / T`; the paper reports results after temperature scaling.

Package: `redeg.fit_temperature` · Notebook: Cell 6.4g.

## 8. Attribution (Eq. 3)

    φ = (c_src(0) − c_src(s5, σ5)) / (c_src(0) − c_tgt),

with `c` the mean maximum softmax probability after temperature scaling; undefined when `|c_src(0) − c_tgt| < 0.005`. `φ₇`
uses the storage-matched source `D(x)` in the numerator. Under the blur hypothesis with `ŝ = s₀` and no other shift `φ = 1`;
with `ŝ = 0`, `φ = 0`. For evaluation only, the labelled share replaces confidences by accuracies (cells with a labelled drop
above 2 pp). **Switch:** use `X@ŝ` when `φ ≥ 0.5`, else `X`. **Blend** (reported): `X + clip(φ, 0, 1) (X@ŝ − X)`.

Package: `redeg.attribution_score`, `redeg.labelled_share`, `redeg.switch`, `redeg.blend` · Notebook: Cells 6.4g, 6.4k.

## 9. Plug-in calibration

For an estimator X calibrated on labelled source data, `X@ŝ` repeats the calibration on the re-degraded source:

| Estimator | Calibration on the source | `X@ŝ` |
|---|---|---|
| ATC, ATC-NE (Garg et al., ICLR 2022) | score threshold at the source error rate | threshold refitted on the re-degraded source |
| DoC (Guillory et al., ICCV 2021) | source accuracy and mean confidence | offset refitted |
| COTT (Lu et al., NeurIPS 2023) | cost threshold with the true source labels | threshold refitted |
| COT (Lu et al., NeurIPS 2023) | none (transport cost) | `COT + [b(s5, σ5) − b(0, 0)]`, `b` = transport cost minus error on the re-degraded source |
| AE-LS (Deng and Zheng, CVPR 2021), MaNo (Xie et al., NeurIPS 2024) | regression on meta-sets | bias transfer `X − [b(s5, σ5) − b(0, 0)]` |

Every form returns X exactly when the re-degraded source is the clean source (rules L1, Q1, W1 check this on every cell).

Package: `redeg.estimators`, `redeg.plugins_at_level`, `redeg.bias_transfer` · Notebook: Cells 3.3, 4.1, 4.2, 4.4
(`plugin_family_v5`), 6.4g, 6.4k (`predict_v7`).

## 10. Floor-sensitivity index (Eq. 4, addendum v7)

    FSI = (1/3) Σ_{s ∈ {1, 4, 16}} 100 [ g(T_s P + 3 DN) − g(T_s P) ],

with `g` the confidence-accuracy gap of the classifier on its labelled source, blurred to `s` in floating point, with and
without 3 DN of white noise. `FSI ≥ 4 pp` predicts that a floating-point re-degradation will raise the estimators' error.

Package: `redeg.floor_sensitivity_index`, `redeg.fsi_inputs`, `redeg.predicts_harm` · Notebook: Cell 6.4k
(`floor_sensitivity`), Cell 6.4m (`fsi_from_logits`, the B27 registration record).

## What the package does not contain

The package holds the method. The experimental machinery stays in the notebook: data acquisition and splits, the frozen
encoders and classifiers, the AutoEval meta-sets (whose regressions AE, AE-LS, AE-FD, NucNorm and MaNo need), SAC-ATC, the
identification baselines (feature regressor, variance of the Laplacian, spectral slope, KernelGAN), the statistics engine
and every figure and table.

Two implementation details differ from the notebook and are documented in the code: `redegrade` draws the noise `ξ` from a
keyed NumPy stream and blurs in float64, while the notebook draws it from a keyed torch generator and blurs in float32 on the
GPU, so individual noise realisations and a few rounding ties differ; and the package assumes every source label belongs to
the classifier's classes (the notebook masks other labels). Everything else is the notebook's code, and
`scripts/check_notebook_equivalence.py` verifies that the two agree to the last bit.
