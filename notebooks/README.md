# Notebooks

The experiments are one self-contained Kaggle notebook, extended once per registration document and run nine times on a
single NVIDIA T4. Each run executed the notebook as registered at that point, restored every finished unit from the
checkpoints of the previous run and computed only the new blocks. **Folder `EXPk` is run k of the paper**: the executed
notebook with all its outputs, and the `outputs.zip` that the run wrote (unit checkpoints, figures, tables, key results,
manifest and log). **`EXP9/EXP9_v8_final.ipynb` produced every number, table and figure of the paper**; the earlier folders
are the record of what each run computed and saw.

| Folder | Run | Session (UTC) | Registration in force | Computed in this run |
|---|:---:|---|---|---|
| [`EXP1`](EXP1) | 1 | 2026-10-11 | v1.0 | the frozen v1.0 programme in one session; its B1 and 16 of the 17 B5 units were reused by run 2 |
| [`EXP2`](EXP2) | 2 | 2026-10-12 | v1.0 + v2 | B2-B4 and B6: LandCover.ai physics, ten datasets x six encoders, real shifts, seeds 1-2; one B5 unit (VICReg) |
| [`EXP3`](EXP3) | 3 | 2026-10-13 | + v3 | B9-B11: ImageNet-C corruptions on the ten datasets, no-reference identification baselines, SpaceNet MVOI |
| [`EXP4a`](EXP4a) | 4, first attempt | 2026-10-14 | + v4 | nothing: the CIFAR-10 data could not be downloaded, so B12 was reported missing; superseded by EXP4b and not used |
| [`EXP4b`](EXP4b) | 4 | 2026-10-14 | + v4 | B12: CIFAR-10-C with six CIFAR-10 classifiers |
| [`EXP5`](EXP5) | 5 | 2026-10-16 | + v5 | B13-B17: CIFAR-100-C, fMoW, the noise semigroup, CIFAR-10-C with the v5 plug-ins, real pairs |
| [`EXP6`](EXP6) | 6 | 2026-10-17 | + v6 | B19-B23: KernelGAN; CIFAR-C, fMoW and real pairs re-run with temperature scaling; B18 was left for run 7 when the session reached its 11-hour budget |
| [`EXP7`](EXP7) | 7 | 2026-10-18 | v6 | B18: ImageNet with the floating-point plug-in |
| [`EXP8`](EXP8) | 8 | 2026-10-19 | + v7 | B24-B26: ImageNet with the storage step, swapped halves, storage-matched real pairs (two sessions; B26 in the first) |
| [`EXP9`](EXP9) | 9 | 2026-10-21 | + v8 | B27: eight unseen ImageNet classifiers, registered before their targets were read; then every analysis, figure and table |

Every folder holds `EXPk_<registration>[_tag].ipynb` and `EXPk_<registration>[_tag]_outputs.zip`; `EXP3` also holds the
Kaggle log of its session (`EXP3_v3.log`). What "registration in force" means and the hashes of each document:
[`../prereg`](../prereg). The block list with the rule of each block is in the introduction of every notebook and in
Supplement C of the paper.

## Reading the record

- **Sessions.** The notebooks count the sessions of the checkpoint chain, which starts with run 2: the line *Key results (full
  run; session of …, N session(s) so far)* reads 1 in run 2, 2 in run 3 and so on up to 6 in run 7. Run 8 ran in two sessions
  on 19 October: the first (18:20 UTC) computed the three B26 units and was restarted 34 minutes later; the second (18:54 UTC)
  is the notebook in `EXP8`, which restores those units from the first session's checkpoints and reports 8 sessions. Run 9
  reports 9. The sessions of the final chain are listed in [`../results/run_manifest.json`](../results/run_manifest.json)
  (`sessions`, and `block_status` for the session in which every unit was computed).
- **Attempts outside the chain.** `EXP4a` is kept because it is part of the record; nothing it wrote was used. A first
  attempt at run 9 (20 October) was stopped by hand while it read the ImageNet-C archive, before block B27 had computed
  anything; it left no checkpoint and its notebook is not included. The reading step was rewritten as a single streaming pass
  before the second attempt, and the change is listed among the notebook's self-selected decisions.
- **Self-check.** Runs 1-6 ran the reproducibility self-check (bit-identical pixels, identical probe predictions, identical
  random streams, frozen encoder) at the start of the session. Runs 7-9 skipped it, as their notebooks print, because their
  remaining blocks prepare the ImageNet data only when the block starts; they rely on the check of run 6 and on the
  self-tests of the addenda they ran.
- **Hashes.** Every session recomputed the hashes of the registration documents at its start, and every session that
  finished recomputed them again in its final cell; all matched.

## What was removed from the exported files

Every notebook here was written by `scripts/prepare_notebooks.py` from the file Kaggle exported. The script keeps every
source and every result and removes two things: the Kaggle, Papermill and Colab metadata that Kaggle stores in the file
(dataset ids, timestamps), and, in the v2 and v3 notebooks, the download link in the last cell that embedded a second copy
of `outputs.zip` as a data URI (20-29 MB; the zip itself is in the same folder). The log is exactly as the run wrote it, and
so is every `outputs.zip`, except that some zips are stored without their figure files (PDF and PNG) to fit GitHub's 25 MiB
upload limit; every entry they keep is byte-identical, and the final run's figures are in [`../figures`](../figures).

```bash
python scripts/prepare_notebooks.py ~/Downloads/*.ipynb --out notebooks
python scripts/check_anonymity.py notebooks        # add --names-file to also search for your own names and handles
```

Nothing in a notebook is executed by either script.

## Running the final notebook

On Kaggle: *File → Import notebook* (`EXP9/EXP9_v8_final.ipynb`), *Accelerator: GPU T4*, *Internet: on*, *Save & Run All*.
No credentials, tokens or logins are used; every dataset and model is downloaded from its public source. The notebook keeps a
universal 11-hour session budget, checkpoints every finished unit, and writes an `outputs.zip` at the end of each session;
attaching that zip as a dataset to the next session resumes the programme where it stopped (attaching a run's
`EXPk_..._outputs.zip` resumes from that run). From scratch the units of the final chain take about 40 hours on one T4, so
expect several sessions. Details, outputs and a guide to the printed log: [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

The method itself does not need the notebook: [`../src/redeg`](../src/redeg) contains it as a small tested package, and
`scripts/check_notebook_equivalence.py --notebook notebooks/EXP9/EXP9_v8_final.ipynb` confirms that package and notebook
compute identical values.
