# Figures

`pdf/` holds the 41 vector figures exactly as the final run wrote them (byte-identical to its `outputs.zip`); every one of them
is in the paper or the supplement. `png/` holds 150-dpi previews rendered from those PDFs for browsing. Which figure is which
in the paper: [`results/README.md`](../results/README.md#paper--file).

The notebook draws each figure from its checkpoints and redraws it from its stored data as a check; all 41 regenerated
identically in the final run (`standalone_regeneration` in [`results/run_manifest.json`](../results/run_manifest.json)).

| | |
|:---:|:---:|
| <img src="png/fig1_overview.png" width="440"><br>Fig. 1 · Overview of the evidence | <img src="png/fig2_physics.png" width="440"><br>Fig. 3 · Physics on real orthophotos |
| <img src="png/fig22_floor.png" width="440"><br>Fig. 4 · The storage floor | <img src="png/fig17_identification.png" width="440"><br>Fig. S4 · Identification on blur the method did not generate |
| <img src="png/fig24_attribution.png" width="440"><br>Fig. S18 · Attribution without labels | <img src="png/fig23_imagenet_v7.png" width="440"><br>Fig. S38 · ImageNet with the storage-matched plug-in |
| <img src="png/fig21_kernelgan.png" width="440"><br>Fig. S7 · Against blind kernel estimation | <img src="png/figS12_gallery.png" width="440"><br>Fig. S15 · The blur ladder on a real tile |
