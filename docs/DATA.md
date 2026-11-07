# Data and models

Every dataset and model is public and is downloaded by the notebook from the source below; nothing is redistributed in this
repository. Licences are as recorded by the notebook at download time (Table S2, [`T11_datasets.csv`](../results/tables/T11_datasets.csv));
check them before any redistribution. Several datasets are for non-commercial use only.

## Datasets

| Dataset | Used for | Source | Licence (as recorded) |
|---|---|---|---|
| RESISC45 | ten-dataset evaluation, real pairs | [torchgeo/resisc45](https://huggingface.co/datasets/torchgeo/resisc45) | CC BY-NC 4.0 (dataset card) |
| UC Merced | ten-dataset evaluation, real pairs | [torchgeo/ucmerced](https://huggingface.co/datasets/torchgeo/ucmerced) | CC0 1.0 (dataset card) |
| PatternNet | ten-dataset evaluation, real pairs | [torchgeo/PatternNet](https://huggingface.co/datasets/torchgeo/PatternNet) | CC BY 4.0 (dataset card) |
| WHU-RS19 | ten-dataset evaluation | [jonathan-roberts1/WHU-RS19](https://huggingface.co/datasets/jonathan-roberts1/WHU-RS19) | CC BY 4.0 (dataset card) |
| RSSCN7 | ten-dataset evaluation | [jonathan-roberts1/RSSCN7](https://huggingface.co/datasets/jonathan-roberts1/RSSCN7) | "other" (dataset card) |
| Optimal-31 | ten-dataset evaluation | [jonathan-roberts1/Optimal-31](https://huggingface.co/datasets/jonathan-roberts1/Optimal-31) | "other" (dataset card) |
| CLRS | ten-dataset evaluation | [jonathan-roberts1/CLRS](https://huggingface.co/datasets/jonathan-roberts1/CLRS) | "other" (dataset card) |
| Imagenette | ten-dataset evaluation | [fastai/imagenette](https://github.com/fastai/imagenette) | see the source |
| DTD | ten-dataset evaluation | [VGG, Oxford](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | see the source |
| Oxford-IIIT Pets | ten-dataset evaluation | [VGG, Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/) | see the source |
| LandCover.ai v1 | physics (knee law, 25 vs 50 cm), real 25 → 50 cm shift, KernelGAN comparison | [landcover.ai](https://landcover.ai.linuxpolska.com/) | CC BY-NC-SA 4.0 |
| SpaceNet MVOI (Atlanta) | 27 off-nadir views | public AWS bucket `spacenet-dataset` (SN4) | CC BY-SA 4.0 |
| fMoW-rgb (validation) | real ground-sample-distance shift, KernelGAN comparison | public AWS bucket `spacenet-dataset/Hosted-Datasets/fmow` | the LICENSE distributed with fMoW-rgb |
| CIFAR-10-C, CIFAR-100-C | official corruption files | Zenodo records [2535967](https://zenodo.org/records/2535967) and [3555552](https://zenodo.org/records/3555552) (md5-checked) | CC BY 4.0 |
| CIFAR-10, CIFAR-100 test sets | clean sources of the CIFAR benchmarks | [cs.toronto.edu/~kriz](https://www.cs.toronto.edu/~kriz/cifar.html) | see the source |
| ImageNetV2 (matched frequency) | ImageNet labelled source and generated targets | [vaishaal/ImageNetV2](https://huggingface.co/datasets/vaishaal/ImageNetV2) | see the source |
| ImageNet-C blur archive | real ImageNet-C targets (three disjoint image sets) | Zenodo record [2235448](https://zenodo.org/records/2235448), `blur.tar` (md5-checked) | see the source |

The ImageNet-C corruption definitions applied to the ten datasets and to ImageNetV2 were re-implemented in NumPy/SciPy and
reproduce the reference package bit for bit in 45 of 45 checks (Cell 5.6).

## Models

| Role | Models | Source |
|---|---|---|
| frozen encoders with linear probes (ten datasets) | DINOv2-S/14, Barlow Twins R50, MAE-B/16, CLIP-B/16, supervised ResNet-50, Satlas Swin-V2-B | official releases via torch.hub, timm and the Hugging Face hub (Cell 6.1) |
| mechanism screen (ledger only) | 17 encoders: the six above and DINOv2-B/14, DINO-S/16, SwAV R50, VICReg R50, DeiT III-S/B, MAE-L/16, CLIP-L/14, DeiT-S/B, I-JEPA-H/14 | as above |
| CIFAR-10 / CIFAR-100 classifiers | ResNet-20, ResNet-56, VGG-16-BN, MobileNetV2 x1.4, ShuffleNetV2 x2.0, RepVGG-A2 | torch.hub `chenyaofo/pytorch-cifar-models` (BSD-3-Clause) |
| ImageNet classifiers (runs 7-8) | `resnet50.tv2_in1k`, `convnext_tiny.fb_in1k`, `swin_tiny_patch4_window7_224.ms_in1k`, `deit_small_patch16_224.fb_in1k` | timm, weights from the Hugging Face hub |
| ImageNet classifiers held out until run 9 | `convnext_small.fb_in1k`, `swin_small_patch4_window7_224.ms_in1k`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`, `deit_base_patch16_224.fb_in1k`, `efficientnet_b0.ra_in1k`, `regnety_040.ra3_in1k`, `mobilenetv3_large_100.ra_in1k`, `resnet50.a1_in1k` | timm, weights from the Hugging Face hub |

Every model is used frozen, in evaluation mode; a parameter checksum before and after inference is part of the
reproducibility self-check.
