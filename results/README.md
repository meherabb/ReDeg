# Results of the final run

Everything in this folder was written by the final notebook (run 9, `notebooks/EXP9/EXP9_v8_final.ipynb`) and is
byte-identical to the files in the `outputs.zip` of that run (`notebooks/EXP9/EXP9_v8_final_outputs.zip`). Every number, table and figure of the paper is computed from these files; the
figures themselves are in [`../figures`](../figures).

| File | Content |
|---|---|
| [`key_results.txt`](key_results.txt) | the run's summary: registration hashes, the five claims with their tests and documented limits, the B27 registration record, the floor-sensitivity index of every ImageNet classifier, the storage read from the files, the v2-v8 headlines, every verdict, benchmark summaries |
| [`tables/`](tables) | the 46 tables of the paper and the supplement as CSV (`T1`-`T45`, the ledger in three parts `T33a`-`T33c`) |
| [`run_manifest.json`](run_manifest.json) | configuration, the eight registration documents and their hashes, environment and package versions, the nine sessions, per-block status and timing, every verdict, the claims maps, self-selected and runtime decisions |
| [`logs/run_log.txt`](logs/run_log.txt) | the full log of the final session, including the restored results of runs 1-8 |

**Units.** In the CSV files accuracies, MAEs and confidence-accuracy gaps are stored as fractions (0.0297 = 2.97 pp), also
where a column or row label says "pp"; floor-sensitivity indices (`fsi*`) are in pp; blur levels are in px²; `rho` columns are
Spearman correlations. Estimator names follow the notebook: `X_v5` / `X_v6` is the floating-point plug-in X@ŝ, `X_v7` the
storage-matched one, `X_sw` / `X_sw7` the switch, `_m` and `_s` the v1.0 plug-in columns.

**Checkpoints.** The 90 per-unit checkpoints of the run (pickle files, read only by the notebook's resume step) are in the
run's `outputs.zip`, not unpacked here; the notebooks restore every result from them, and the tables here are what they
produce. The per-image logits are not included.

## Paper → file

Figure and table numbers are those of the submitted paper. The introduction of the notebook contains a "paper map" written
before the paper (addendum v8); the numbering below supersedes it.

### Main paper

| Paper | Content | Files |
|---|---|---|
| Fig. 1 | Overview: identify, attribute, diagnose, calibrate | [`fig1_overview.pdf`](../figures/pdf/fig1_overview.pdf) |
| Fig. 2 | Pipeline | TikZ in the paper; [`assets/pipeline.png`](../assets/pipeline.png) |
| Fig. 3 | Physics on real LandCover.ai orthophotos | [`fig2_physics.pdf`](../figures/pdf/fig2_physics.pdf) |
| Fig. 4 | The storage floor | [`fig22_floor.pdf`](../figures/pdf/fig22_floor.pdf) |
| Table 1 | Identification against a physical truth | [`T35_identification.csv`](tables/T35_identification.csv) |
| Table 2 | ImageNet with storage matching | [`key_results.txt`](key_results.txt) (block "Addenda v7 and v8 headline"); per set: [`T36_imagenet_storage.csv`](tables/T36_imagenet_storage.csv), [`T45_unseen_mae.csv`](tables/T45_unseen_mae.csv) |
| Table 3 | Floor sensitivity of the twelve ImageNet classifiers | [`T41_floor_unseen.csv`](tables/T41_floor_unseen.csv); registered values: [`prereg/b27_registration_record.json`](../prereg/b27_registration_record.json) |
| Table 4 | Registered claims | [`T42_claims_v8.csv`](tables/T42_claims_v8.csv) |

### Supplement: figures

| Paper | Content | File |
|---|---|---|
| Fig. S1 | Checks of the theory on the exact operators | [`figS1_theory_checks.pdf`](../figures/pdf/figS1_theory_checks.pdf) |
| Fig. S2 | The test-inversion certificate | [`figS2_certificate.pdf`](../figures/pdf/figS2_certificate.pdf) |
| Fig. S3 | The pipeline as registered in v1.0 and extended in v2 | [`fig4_method.pdf`](../figures/pdf/fig4_method.pdf) |
| Fig. S4 | Identification on blur the method did not generate | [`fig17_identification.pdf`](../figures/pdf/fig17_identification.pdf) |
| Fig. S5 | Identification baselines | [`figS18_id_baselines.pdf`](../figures/pdf/figS18_id_baselines.pdf) |
| Fig. S6 | Which identifier, and a second semigroup | [`fig15_ablation_noise.pdf`](../figures/pdf/fig15_ablation_noise.pdf) |
| Fig. S7 | Identification against blind kernel estimation (KernelGAN) | [`fig21_kernelgan.pdf`](../figures/pdf/fig21_kernelgan.pdf) |
| Fig. S8 | fMoW, a real change of GSD | [`fig16_fmow.pdf`](../figures/pdf/fig16_fmow.pdf) |
| Fig. S9 | SpaceNet MVOI off-nadir views | [`fig10_mvoi.pdf`](../figures/pdf/fig10_mvoi.pdf) |
| Fig. S10 | The real 25 cm → 50 cm change of LandCover.ai | [`fig8_real_gsd.pdf`](../figures/pdf/fig8_real_gsd.pdf) |
| Fig. S11 | Sensitivity of identification | [`figS6_identification_sensitivity.pdf`](../figures/pdf/figS6_identification_sensitivity.pdf) |
| Fig. S12 | Stress tests | [`figS13_stress_tests.pdf`](../figures/pdf/figS13_stress_tests.pdf) |
| Fig. S13 | The noise-aware path | [`figS16_noise_v2.pdf`](../figures/pdf/figS16_noise_v2.pdf) |
| Fig. S14 | Where the guarantees end | [`fig6_limits.pdf`](../figures/pdf/fig6_limits.pdf) |
| Fig. S15 | The blur ladder on a real tile | [`figS12_gallery.pdf`](../figures/pdf/figS12_gallery.pdf) |
| Fig. S16 | One target image per dataset | [`figS15_dataset_gallery.pdf`](../figures/pdf/figS15_dataset_gallery.pdf) |
| Fig. S17 | ImageNet-C corruptions on three datasets | [`figS17_corruption_gallery.pdf`](../figures/pdf/figS17_corruption_gallery.pdf) |
| Fig. S18 | Attribution without labels | [`fig24_attribution.pdf`](../figures/pdf/fig24_attribution.pdf) |
| Fig. S19 | Attribution and the switch | [`fig19_attribution.pdf`](../figures/pdf/fig19_attribution.pdf) |
| Fig. S20 | The frozen v1.0 evaluation | [`fig5_main_results.pdf`](../figures/pdf/fig5_main_results.pdf) |
| Fig. S21 | Re-blurring a real tile, and the first five runs at a glance | [`fig1_teaser.pdf`](../figures/pdf/fig1_teaser.pdf) |
| Fig. S22 | Statistics and seeds | [`figS3_statistics_seeds.pdf`](../figures/pdf/figS3_statistics_seeds.pdf) |
| Fig. S23 | Encoder breakdown | [`figS4_encoder_breakdown.pdf`](../figures/pdf/figS4_encoder_breakdown.pdf) |
| Fig. S24 | Estimator ablation | [`figS5_estimator_ablation.pdf`](../figures/pdf/figS5_estimator_ablation.pdf) |
| Fig. S25 | Accuracy against cost | [`figS7_efficiency.pdf`](../figures/pdf/figS7_efficiency.pdf) |
| Fig. S26 | Accuracy curves | [`figS8_accuracy_curves.pdf`](../figures/pdf/figS8_accuracy_curves.pdf) |
| Fig. S27 | Predictions by regime | [`figS11_prediction_scatter.pdf`](../figures/pdf/figS11_prediction_scatter.pdf) |
| Fig. S28 | Real shifts | [`figS9_real_shifts.pdf`](../figures/pdf/figS9_real_shifts.pdf) |
| Fig. S29 | Re-degradation as a plug-in | [`fig7_plugin.pdf`](../figures/pdf/fig7_plugin.pdf) |
| Fig. S30 | Regime map | [`figS14_regime_map.pdf`](../figures/pdf/figS14_regime_map.pdf) |
| Fig. S31 | The exploratory plug-in matrix | [`fig12_plugin_matrix.pdf`](../figures/pdf/fig12_plugin_matrix.pdf) |
| Fig. S32 | ImageNet-C corruptions on the ten datasets | [`fig9_corruptions.pdf`](../figures/pdf/fig9_corruptions.pdf) |
| Fig. S33 | CIFAR-10-C | [`fig11_cifar10c.pdf`](../figures/pdf/fig11_cifar10c.pdf) |
| Fig. S34 | CIFAR-100-C | [`fig14_cifar100c.pdf`](../figures/pdf/fig14_cifar100c.pdf) |
| Fig. S35 | The identified level as a plug-in, addenda v2-v5 | [`fig13_plugin_headline.pdf`](../figures/pdf/fig13_plugin_headline.pdf) |
| Fig. S36 | Temperature scaling | [`fig18_calibration.pdf`](../figures/pdf/fig18_calibration.pdf) |
| Fig. S37 | ImageNet with the floating-point plug-in | [`fig20_imagenet.pdf`](../figures/pdf/fig20_imagenet.pdf) |
| Fig. S38 | ImageNet with the storage-matched plug-in | [`fig23_imagenet_v7.pdf`](../figures/pdf/fig23_imagenet_v7.pdf) |

### Supplement: tables

| Paper | Content | File |
|---|---|---|
| Table S1 | Numerical checks of the theory (rules R0-R10) | [`T7_physics_theory.csv`](tables/T7_physics_theory.csv) |
| Table S2 | Datasets, sources and licences | [`T11_datasets.csv`](tables/T11_datasets.csv) |
| Table S3 | Temperature scaling per classifier | [`T27_temperatures.csv`](tables/T27_temperatures.csv) |
| Table S4 | Compute per block and encoder (one T4 GPU) | [`T10_compute.csv`](tables/T10_compute.csv) |
| Table S5 | Identification without labels (full version of Table 1) | [`T35_identification.csv`](tables/T35_identification.csv) |
| Table S6 | Storage read from the target files (rule W2) | [`T40_storage.csv`](tables/T40_storage.csv) |
| Table S7 | The v1.0 fit check against the number of target images | [`T44_fitcheck_n.csv`](tables/T44_fitcheck_n.csv) |
| Table S8 | Identification and abstention per dataset | [`T3_identification.csv`](tables/T3_identification.csv) |
| Table S9 | Identification baselines | [`T16_id_baselines.csv`](tables/T16_id_baselines.csv) |
| Table S10 | Identifier ablation | [`T22_identifier_ablation.csv`](tables/T22_identifier_ablation.csv) |
| Table S11 | A second semigroup (block B15) | [`T24_noise_semigroup.csv`](tables/T24_noise_semigroup.csv) |
| Table S12 | Identification against KernelGAN (block B19) | [`T30_kernelgan.csv`](tables/T30_kernelgan.csv) |
| Table S13 | fMoW per GSD bin (block B14) | [`T23_fmow.csv`](tables/T23_fmow.csv) |
| Table S14 | SpaceNet MVOI per view (block B11) | [`T15_mvoi.csv`](tables/T15_mvoi.csv) |
| Table S15 | Frozen v1.0 main results per dataset | [`T1_main_results.csv`](tables/T1_main_results.csv) |
| Table S16 | Paired inference for ATC@ŝ against each baseline | [`T2_inference.csv`](tables/T2_inference.csv) |
| Table S17 | Seed stability | [`T6_seeds.csv`](tables/T6_seeds.csv) |
| Table S18 | Ablations | [`T5_ablations.csv`](tables/T5_ablations.csv) |
| Table S19 | Comparison with the strongest baseline measured in this study | [`T9_sota.csv`](tables/T9_sota.csv) |
| Table S20 | Plug-in re-degradation (addendum v2) | [`T13_plugin.csv`](tables/T13_plugin.csv) |
| Table S21 | Regime map | [`T12_regimes.csv`](tables/T12_regimes.csv) |
| Table S22 | Plug-in matrix (exploratory) | [`T19_plugin_matrix.csv`](tables/T19_plugin_matrix.csv) |
| Table S23 | ImageNet-C corruptions on the ten datasets (block B9) | [`T14_corruptions.csv`](tables/T14_corruptions.csv) |
| Table S24 | CIFAR-10-C (block B12) | [`T18_cifar10c.csv`](tables/T18_cifar10c.csv) |
| Table S25 | CIFAR-100-C (block B13) | [`T20_cifar100c.csv`](tables/T20_cifar100c.csv) |
| Table S26 | The v5 plug-in matrix | [`T21_plugin_matrix_v5.csv`](tables/T21_plugin_matrix_v5.csv) |
| Table S27 | Headline of addenda v2-v5 | [`T25_headline.csv`](tables/T25_headline.csv) |
| Table S28 | The v6 plug-in matrix after temperature scaling | [`T28_plugin_matrix_v6.csv`](tables/T28_plugin_matrix_v6.csv) |
| Table S29 | Headline after temperature scaling (addendum v6) | [`T34_headline_v6.csv`](tables/T34_headline_v6.csv) |
| Table S30 | ImageNet with the floating-point plug-in (block B18) | [`T29_imagenet.csv`](tables/T29_imagenet.csv) |
| Table S31 | Floor sensitivity of the four classifiers of runs 7-8 | [`T39_floor.csv`](tables/T39_floor.csv) |
| Table S32 | ImageNet with the storage-matched plug-in | [`T36_imagenet_storage.csv`](tables/T36_imagenet_storage.csv) |
| Table S33 | Eight unseen ImageNet classifiers (block B27) | [`T45_unseen_mae.csv`](tables/T45_unseen_mae.csv) |
| Table S34 | The registration documents and their hashes | [`prereg/`](../prereg) |
| Table S35 | Claims map of addendum v8 | [`T42_claims_v8.csv`](tables/T42_claims_v8.csv) |
| Table S36 | Claims map of addendum v7 | [`T37_claims_v7.csv`](tables/T37_claims_v7.csv) |
| Table S37 | Claims map of addendum v6 | [`T32_claims.csv`](tables/T32_claims.csv) |
| Table S38 | Ledger of all registered rules | [`T33a_ledger.csv`](tables/T33a_ledger.csv), [`T33b_ledger.csv`](tables/T33b_ledger.csv), [`T33c_ledger.csv`](tables/T33c_ledger.csv) |
| Table S39 | Verdicts of v1.0 and v2 | [`T8_verdicts.csv`](tables/T8_verdicts.csv) |
| Table S40 | Verdicts of addenda v3 and v4 | [`T17_verdicts_v3.csv`](tables/T17_verdicts_v3.csv) |
| Table S41 | Verdicts of addendum v5 | [`T26_verdicts_v5.csv`](tables/T26_verdicts_v5.csv) |
| Table S42 | Verdicts of addendum v6 | [`T31_verdicts_v6.csv`](tables/T31_verdicts_v6.csv) |
| Table S43 | Verdicts of addendum v7 | [`T38_verdicts_v7.csv`](tables/T38_verdicts_v7.csv) |
| Table S44 | Verdicts of addendum v8 | [`T43_verdicts_v8.csv`](tables/T43_verdicts_v8.csv) |
