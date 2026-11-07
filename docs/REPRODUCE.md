# Reproducing the paper

There are three levels of reproduction, from seconds to days.

| Level | What it checks | Command | Time |
|---|---|---|---|
| 1 | the registration documents and the B27 record hash to the published values | `python scripts/verify_prereg.py` | 1 s |
| 2 | the package computes what the notebook computed (61 quantities, bit-identical), the theory checks, and block B1's published half-energy table | `python -m pytest` and `python scripts/check_notebook_equivalence.py --notebook notebooks/EXP9/EXP9_v8_final.ipynb` | under a minute, CPU |
| 3 | every number, table and figure, from the data | run `notebooks/EXP9/EXP9_v8_final.ipynb` (below) | about 40 GPU-hours from scratch on one T4 |

## Running the final notebook

**Kaggle (as in the paper).** *File → Import notebook* → `notebooks/EXP9/EXP9_v8_final.ipynb`; *Settings*: Accelerator *GPU T4*
(one GPU is used even on *T4 x2*), Internet *on*; *Save & Run All*. Every dataset and model is downloaded from its public
source ([`DATA.md`](DATA.md)); no credentials, tokens or logins are used.

**Sessions and resuming.** The notebook keeps one session budget (11 h; no new unit starts in the last 0.25 h, packaging
always runs), checkpoints every finished unit, and ends each session with `outputs.zip` in `/kaggle/working`. To continue,
upload that zip as a Kaggle dataset and attach it (*Add Input*), then run all cells again: every checkpoint under
`/kaggle/input` is found automatically (zip or folder, any name, any depth), finished units are restored, and only the missing
ones run. Cell 1.5 prints the plan of each session (✓ restored, • computed now), and every block cell prints
**↺ RESTORED FROM CHECKPOINT** for a restored unit, followed by its key tables. The `outputs.zip` of every run of the paper
is in its folder, `notebooks/EXPk/EXPk_..._outputs.zip`; attaching one resumes the programme from the end of that run.

**Other environments.** On Colab or a local machine with a GPU the notebook writes to `./outputs` and caches data in
`/tmp/redeg_data`; point `REDEG_RESUME_DIR` at a folder or zip of earlier outputs to resume.

**Run modes** (environment variables read by Cell 1.1):

| Variable | Values | Effect |
|---|---|---|
| `REDEG_MODE` | `full` (default) | the GPU run |
| | `analysis_only` | rebuild every verdict, table and figure on CPU from attached checkpoints |
| `DRY_RUN` | `1` | synthetic smoke test with stand-in models on CPU; outputs go to `outputs_dryrun` |
| `REDEG_BUDGET_H` | hours, default `11` | session budget |
| `REDEG_RESUME_DIR` | path | extra place to look for earlier outputs (outside Kaggle) |

## Compute

The units of the final checkpoint chain, by session (one NVIDIA T4, 4 CPU cores; from `block_status` in
[`../results/run_manifest.json`](../results/run_manifest.json)):

| Run | Folder | Blocks computed | Hours |
|:---:|---|---|---:|
| 2 | `EXP2` | B2-B4, B6 (and one B5 unit) | 7.0 |
| 3 | `EXP3` | B9-B11 | 7.1 |
| 4 | `EXP4b` | B12 | 0.8 |
| 5 | `EXP5` | B13-B17 | 6.1 |
| 6 | `EXP6` | B19-B23 | 6.1 |
| 7 | `EXP7` | B18 | 1.7 |
| 8 | `EXP8` | B24-B26 | 7.3 |
| 9 | `EXP9` | B27, analysis and packaging | 4.1 |
| | | **total** | **40.1** |

Run 1 contributed the B1 theory checks and 16 of the 17 B5 mechanism units (about 0.9 h). Per-block and per-encoder
timings are in [`T10_compute.csv`](../results/tables/T10_compute.csv) (Table S4). The largest downloads are the ImageNet-C blur
archive (7.1 GB) and ImageNetV2 (1.3 GB).

## Environment of the final run

Python 3.13.15, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1, pandas 2.3.3, Matplotlib 3.10.0, PyTorch 2.11.0 (CUDA 12.8),
torchvision 0.26.0, timm 1.0.29, POT 0.9.7.post1, Pillow 12.3.0, huggingface_hub 1.29.0, datasets 4.8.5; one Tesla T4. The
notebook installs what is missing in its first cell; the exact environment of every session is recorded in
`run_manifest.json` (`environment`).

## Determinism

Every random draw comes from a keyed stream, `numpy.random.default_rng(SeedSequence([0, md5(key)...]))`, never from a global
seed, so any unit can be recomputed in any order and session (`redeg._keys.rng_for` is the same construction). cuBLAS runs in
deterministic mode on one GPU. Before the first GPU unit of a session, a reproducibility self-check (Cell 6.4f) runs the pixel
identification twice (bitwise identical), extracts features twice, redraws the keyed random streams and checks that the
frozen encoder's parameters did not change; the run stops if any check fails. Runs 1-6 ran it. Runs 7-9 skipped it, as their
notebooks print, because their remaining blocks prepare the ImageNet data only when the block starts; they rely on the check
of run 6 and on the self-tests of the addenda they ran. Before the outputs are packed, every figure is
redrawn from its stored data and compared with the original (all 41 identical in the final run).

## Reading the outputs

The `outputs.zip` of the final run holds the 90 unit checkpoints, the 41 figures (PDF and PNG), the 46 tables (CSV and TeX),
`key_results.txt`, the anonymous `run_manifest.json`, `logs/run_log.txt` and a note on resuming; per-image logits are written
to a `logits` folder next to it. That zip is [`notebooks/EXP9/EXP9_v8_final_outputs.zip`](../notebooks/EXP9); its tables,
figures, key results, manifest and log are also unpacked in [`../results`](../results) and [`../figures`](../figures), and
[`../results/README.md`](../results/README.md) maps each file to the paper. The logits are not included.
