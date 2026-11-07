# Registration documents

The method, its constants and its decision rules were written down before the evaluation and extended seven times. Each
document was hashed and stored before the run that first tested it, and the notebook recomputed every hash at the start of
every session and at the end of every session that finished (Cell 1.4 and the final cell). This folder holds the eight documents exactly as the final run
embedded them, the B27 registration record, and the registered hashes.

```bash
python scripts/verify_prereg.py      # standard library only; prints ALL HASHES MATCH
```

## Documents

| Document | Frozen | Rules | What it adds | SHA-256 |
|---|---|---:|---|---|
| [`v1.0.json`](v1.0.json) | before run 1 | 24 | frozen method and constants; theory and physics (R0–R10), method (F1–F9), mechanism (M1–M4) | `58594e8c…b6aebfb5` |
| [`v2.json`](v2.json) | after run 1, before run 2 | 8 | noise-aware path, the plug-ins X@ŝ, real-shift pairs, GSD-aware path (P1–P4, N1–N3, G1) | `9e3ab085…1c863998` |
| [`v3.json`](v3.json) | after run 2, before run 3 | 12 | ImageNet-C corruptions on the ten datasets, no-reference identification baselines, SpaceNet MVOI (C1–C6, V1–V4, I1–I2) | `86ab550b…07420faf` |
| [`v4.json`](v4.json) | after run 3, before run 4 | 4 | official CIFAR-10-C with six CIFAR-10 classifiers (K1–K4) | `b885f994…2639d209` |
| [`v5.json`](v5.json) | after run 4, before run 5 | 13 | identity-preserving plug-ins, identifier ablation, CIFAR-100-C, fMoW, a second semigroup (L1–L13) | `ecec3613…de57fc3c` |
| [`v6.json`](v6.json) | after run 5, before run 6 | 17 | temperature scaling, MaNo, bias transfer, attribution φ and the switch, ImageNet, KernelGAN (Q1–Q17) | `b66b2997…c97d6580` |
| [`v7.json`](v7.json) | after run 7, before run 8 | 18 | storage-matched re-degradation, floor-sensitivity index, ImageNet storage and swapped halves (W1–W18) | `be1de050…8af9b131` |
| [`v8.json`](v8.json) | after run 8, before run 9 | 12 | eight unseen ImageNet classifiers with registered predictions, fit check against *n*, final claims map (U1–U12) | `9dbd90fa…bb027cb2` |

Full hashes are in [`registered_sha256.json`](registered_sha256.json) and in Table S34 of the paper. The counts include the
rules that are reported without a threshold; together with L-sum, a summary row of the v5 verdict table, they make the 109
rules of the paper (54 pass, 13 partial, 14 fail, 28 reported). The JSON of v1.0 holds the thresholded rules; the two reported
analyses F8 and F9 are defined in the registered specification that the notebook restates verbatim in its Section 0.

## How the hash is computed

```python
hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
```

The files here are pretty-printed for reading; the hash is defined on the canonical serialisation above, so reformatting a file
does not change it, while changing any key or value does.

## The B27 registration record

Addendum v8 required the floor-sensitivity index (FSI) of eight ImageNet classifiers that no run had used, and its prediction,
to be recorded before any of their target images was read. The final run wrote
[`b27_registration_record.json`](b27_registration_record.json) at `2026-10-21T00:53:13`, fifteen minutes into the session
(`before_targets: true`). Its hash covers the index of each classifier (rounded to six decimals) and the prediction:

```
e7c59048e412a5a376e40d5fbc241323e264ba02abfd1979bc09c583bd57078b
```

The second stage of block B27 recomputed every registered index from the classifier banks; the largest difference was
0.00 pp (rule U1).

## Scope of the registration

The timestamps and hashes are our own record, so the registration is a commitment device that makes later changes visible, not
an external audit. The claims map was revised twice (v7 and v8), each time after some of the verdicts it reassigns were known;
every rule keeps its verdict across the three maps, and the ledger (`results/tables/T33a_ledger.csv` to `T33c_ledger.csv`) gives
all three assignments.
