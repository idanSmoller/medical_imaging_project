# Probabilistic U-Net

This repository contains a generic PyTorch implementation of the
[Probabilistic U-Net](https://arxiv.org/abs/1806.05034) that somewhat mirrors
the signature of the
[official implementation](https://github.com/SimonKohl/probabilistic_unet) in
Tensorflow.

The legacy experiment-management code has been removed so the project can run
with a small, explicit PyTorch training loop suitable for Colab and course
submission.

For the single-notebook submission entry point, use
`notebooks/disagreement_aware_probunet_colab.ipynb`. It is self-contained: the
notebook cells include the downloader, dataset, model, losses, training loop,
evaluation metrics, checkpointing, and qualitative figure generation.


## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

will make a package called `probunet` available in your current Python environment.


## Data: LIDC-IDRI crops

```bash
python scripts/download_lidc.py
```

Downloads (~215 MB) and extracts the preprocessed LIDC-IDRI 2D crops into `data/lidc/`,
then checks the result against the published counts and fails loudly on a mismatch:

| split | images | patients |
| ----- | ------ | -------- |
| train | 8843   | 530      |
| val   | 1993   | 111      |
| test  | 1980   | 103      |

This is DeepMind's release of the crops (CC BY 3.0), published with the
[Hierarchical Probabilistic U-Net](https://github.com/google-deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet)
and linked from the [official Probabilistic U-Net repo](https://github.com/SimonKohl/probabilistic_unet).
Its preprocessing matches Kohl et al. Appendix H.1 (0.5 mm x 0.5 mm in-plane resample,
180 x 180 crops centered on the abnormality, polygon-only lesions > 3 mm, graders matched
by overlapping bounding boxes). The counts differ from the paper's 8882 / 1996 / 1992 by
about 0.4% because the authors cleaned up the data after publication, stating this
"leaves the results the same". No TCIA account or `gsutil` is needed.

Usage:

```python
from probunet.lidc import LIDCCrops

dataset = LIDCCrops(split="train")     # 128x128 random crop, per Appendix H.1
sample = dataset[0]
sample["image"]  # FloatTensor [1, 128, 128], values in [0, 1]
sample["masks"]  # FloatTensor [4, 128, 128], binary, one per grader
```

The `masks` axis is the `G` axis expected by `probunet.disagreement.compute_disagreement`
and `DisagreementAwareProbabilisticSegmentationNet.disagreement_losses`. Pass
`single_random_grader=True` to additionally get `target`, one uniformly drawn grader mask,
which is how the baseline model draws image-grader pairs during training. 65% of training
crops have at least one empty mask, since graders disagree on whether a lesion is present.


## Probabilistic U-Net

Our generic implementation of the Probabilistic U-Net is hopefully relatively straightforward to use:

```python
from probunet.model import ProbabilisticSegmentationNet
```

As you will have noticed, it also has a pretty generic name. That's because it doesn't actually require a U-Net, but can work with arbitrary segmentation architectures, as long as they:

1. Allow injection of samples in some way.
2. Provide the same signature as our InjectionUNet (look at the calls to self.task_net to see requirements.)

Our encoder implementation also accepts injections, this is currently not used. Make sure to read the method docstrings of the ProbabilisticSegmentationNet, there are some quirks, e.g. `.reconstruct()` doesn't compute gradients.

## Baseline Smoke Training

The repository includes a small plain-PyTorch baseline runner:

```bash
python scripts/baseline_smoke_train.py --epochs 3 --device cpu
```

It trains the original Probabilistic U-Net mechanics on the LIDC crops (after
running `scripts/download_lidc.py`), drawing one random grader per image as the
target. It defaults to a small subset (`--max-train` / `--max-val`) and is only a
wiring check, not a result. Pass `--dataset npz --npz PATH` to instead use a local
lesion sequence NPZ with bounding boxes rasterized into masks. It saves:

- `outputs/baseline_smoke/baseline_checkpoint.pt`
- `outputs/baseline_smoke/history.npy`
- `outputs/baseline_smoke/qualitative_samples.png`

For the final disagreement-aware project experiments, prefer adding similarly
explicit PyTorch scripts under `scripts/`.

## LIDC Ablation Training

`scripts/train_lidc_ablation.py` starts the three planned variants:

```bash
python3 scripts/train_lidc_ablation.py --variant baseline
python3 scripts/train_lidc_ablation.py --variant head --lambda-disagreement 0.5
python3 scripts/train_lidc_ablation.py --variant full --lambda-disagreement 0.5 --lambda-alignment 0.5
```

The defaults follow the LIDC Appendix H.1 shape where practical: 128 x 128 crops,
batch size 32, Adam, 1e-4 to 1e-6 stepped learning rate, weight decay 1e-5,
6-D latent, base 32 channels, and 4 prior samples for the alignment loss. The script
logs Dice, IoU, generalized energy distance, disagreement MAE, and disagreement
correlation to `history.csv`, and writes `best_checkpoint.pt` / `latest_checkpoint.pt`
under `outputs/lidc_ablation/<variant>/`.

Resume an interrupted run by pointing `--resume` at the latest checkpoint and keeping
`--steps` as the final global step you want to reach:

```bash
python3 scripts/train_lidc_ablation.py --variant full \
  --resume outputs/lidc_ablation/full/latest_checkpoint.pt \
  --steps 240000
```

Pass `--reset-optimizer` only when you want to load model weights but intentionally
restart Adam state while preserving the global-step learning-rate schedule.

For a fast wiring check:

```bash
python3 scripts/train_lidc_ablation.py --variant full --steps 1 --eval-every 1 \
  --max-train 2 --max-val 2 --batch-size 1 --eval-batch-size 1 \
  --feature-maps 2 --latent-size 2 --depth 2 --train-samples 2 --eval-samples 2 \
  --device cpu --out-dir outputs/lidc_ablation_smoke/full
```

## Qualitative Figures

After training baseline and full checkpoints, generate Phase 10 comparison figures:

```bash
python3 scripts/make_lidc_figures.py \
  --baseline-checkpoint outputs/lidc_ablation/baseline/best_checkpoint.pt \
  --full-checkpoint outputs/lidc_ablation/full/best_checkpoint.pt \
  --split test \
  --num-cases 8 \
  --out-dir outputs/lidc_figures
```

By default the script picks test cases with the highest mean human disagreement and
writes `case_*.png` files containing the input image, four human masks, human
disagreement, baseline samples/uncertainty, full-model samples/uncertainty, and the
full model's predicted disagreement map. Use `--indices 0 10 42` to plot specific
dataset indices instead.
