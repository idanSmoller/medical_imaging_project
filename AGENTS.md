# Agent instructions

Shared context for coding agents (Codex, Claude Code, …) working on this repo.
Two people collaborate here using different agents, so **treat this file as the
handoff channel**: if you learn something the next agent would otherwise
rediscover the hard way, add it here.

## What this project is

A PyTorch Probabilistic U-Net ([Kohl et al., NeurIPS 2018](https://arxiv.org/abs/1806.05034))
extended to be *disagreement-aware*: alongside the segmentation it predicts where
human annotators disagree, and aligns sample diversity with that disagreement.

The full project plan is `plans/disagreement_aware_probabilistic_unet_plan.md`
(12 phases). Model-side phases (3–5) are implemented; see "Current state" below.

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # required — scripts/ import `probunet`, which is not on sys.path otherwise
python scripts/download_lidc.py
```

**If you add or upgrade a dependency, put it in BOTH `requirements.txt` and
`install_requires` in `setup.py`.** The other collaborator's environment is
built independently, so anything installed only in your shell silently breaks
their runs.

## Data

`scripts/download_lidc.py` fetches the preprocessed LIDC-IDRI 2D crops (~215 MB)
into `data/lidc/` (gitignored) and validates the counts, exiting non-zero on a
mismatch:

| split | images | patients |
| ----- | ------ | -------- |
| train | 8843   | 530      |
| val   | 1993   | 111      |
| test  | 1980   | 103      |

Layout: `data/lidc/<split>/images/<patient>/z-<z>_c<n>.png` and
`data/lidc/<split>/gt/<patient>/z-<z>_c<n>_l<grader>.png`. Images are 8-bit
grayscale 180×180; masks are `{0, 255}`.

**Gotcha: grader suffixes on disk are `_l0`–`_l3`, not `_l1`–`_l4` as the
upstream release README claims.** Code that trusts the README silently drops a
grader.

Use `probunet.lidc.LIDCCrops` — do not re-read the PNGs by hand:

```python
from probunet.lidc import LIDCCrops
ds = LIDCCrops(split="train")          # 128x128 random crop (paper, Appendix H.1)
ds[0]["image"]   # FloatTensor [1, 128, 128] in [0, 1]
ds[0]["masks"]   # FloatTensor [4, 128, 128] binary, one per grader
```

`masks` is already the `(B, G, *spatial)` layout that
`probunet.disagreement.compute_disagreement` and
`DisagreementAwareProbabilisticSegmentationNet.disagreement_losses` expect —
**reuse those, don't write parallel metric code.** Pass
`single_random_grader=True` to also get `target`, one uniformly drawn grader mask
(how the baseline draws image-grader pairs). 65% of training crops have ≥1 empty
mask, because graders disagree on whether a lesion is present at all; 0% are
all-empty. That is signal, not corruption.

### Which dataset version, and why it matters for the report

This is DeepMind's release of the crops (CC BY 3.0), published with the
Hierarchical Probabilistic U-Net and linked from the official Probabilistic U-Net
repo. Its preprocessing matches the paper's Appendix H.1 verbatim on four
idiosyncratic points (0.5 mm in-plane resample; polygon-only lesions > 3 mm;
dropping DICOMs where `|SliceLocation| != |ImagePositionPatient[-1]|`; matching
graders by overlapping bounding boxes).

Counts are ~0.4% below the paper's 8882 / 1996 / 1992 because the authors cleaned
up the data after publication, stating it "leaves the results the same". The exact
2018 snapshot was never released. **Cite it as the HPU-Net release and mention the
delta** rather than claiming byte-identical reproduction.

Rejected alternatives, do not switch to these without a reason: the `stefanknegt`
Google Drive pickle (independent third-party preprocessing, no documented split)
and raw DICOM from TCIA (~125 GB; the project plan rules it out).

## Repo conventions

- **The package is flat under `probunet/`.** The nested layout in the plan doc
  (`data/`, `training/`, `evaluation/`, …) was never adopted — do not create it.
- **`probunet/data.py` is dead legacy** — the upstream 3D brain-MRI loader, with
  `data_dir = None`, expecting `(time, channel, x, y, z)` volumes, and importing
  `batchgenerators`, which is not installed. Nothing in the LIDC pipeline uses it.
  Don't "fix" it; use `probunet/lidc.py`.
- Scripts go in `scripts/` as explicit plain-PyTorch programs with argparse. The
  experiment-management framework (`trixi`) was deliberately removed so the project
  runs in Colab.
- `data/`, `outputs/`, `checkpoints/`, `*.pt`, `*.npy`, `*.npz` are gitignored.
  Don't commit weights or data.

## Verification

```bash
python scripts/download_lidc.py                                        # counts must print OK
python scripts/baseline_smoke_train.py --dataset lidc --epochs 1 --device cpu
```

The smoke run is a **wiring check, not a result** — it defaults to 64 train / 32
val images on a tiny 8-feature-map net, so a near-zero val dice is expected. It
writes `outputs/baseline_smoke/qualitative_samples.png`, which shows the image,
target, model samples, and all four grader masks; the graders should visibly
differ.

## Paper reference values (arXiv 1806.05034v4 — appendices; the PDF in this repo is main text only)

- **Appendix H.1 (LIDC training):** 128×128 crops of the 180×180 tiles, batch 32,
  Adam, lr 1e-4 decayed to 1e-6 in 5 steps, 240k iterations, weight decay 1e-5,
  **β = 1 with a 6-D latent** for the Probabilistic U-Net, base 32 channels over
  4 down/up-samplings, orthogonal init (gain 1). Augmentation: elastic, rotation,
  shearing, scaling, then the random translated crop. Only the crop is implemented.
- **Appendix B (metric):** generalized energy distance over n model samples vs.
  m = 4 ground truths with d = 1 − IoU, and **d ≡ 0 when both masks are empty**, so
  agreement on lesion absence is rewarded. This case is common here — get it right.

## Current state

Done: LIDC download + loader, disagreement utilities (`probunet/disagreement.py`),
disagreement head and losses (`probunet/model.py`), smoke training on LIDC,
generalized energy distance, and a first LIDC ablation training script.

Not done: long ablation runs in plan Phase 8 (A baseline / B +head / C full,
λ ∈ {0.1, 0.5, 1.0}); final test-set evaluation and qualitative comparison figures.

## Handoff log

Append dated entries. Keep them short — what changed and what the next agent should know.

- **2026-08-03 (Claude):** Added `scripts/download_lidc.py`, `probunet/lidc.py`,
  `requirements.txt`; pointed `baseline_smoke_train.py` at LIDC through a real
  `DataLoader` and removed a hardcoded absolute NPZ path. Verified counts, tensor
  shapes/ranges, and that disagreement maps are non-trivial. Found the `_l0`–`_l3`
  vs `_l1`–`_l4` discrepancy documented above.
- **2026-08-11 (Codex):** Added `probunet.eval.generalized_energy_distance`
  with Appendix B empty-mask handling, plus `scripts/train_lidc_ablation.py` for
  baseline/head/full LIDC runs. Verified one-step CPU smoke runs for all three
  variants using `PYTHONPATH=.`; for normal use run `pip install -e .` first.
- **2026-08-11 (Codex):** Added `--resume` / `--reset-optimizer` support to
  `scripts/train_lidc_ablation.py`. New checkpoints include Python, NumPy, Torch,
  and CUDA RNG state; smoke-tested resume from step 1 to step 2 on CPU.
- **2026-08-11 (Codex):** Added `scripts/make_lidc_figures.py` for Phase 10
  qualitative grids comparing baseline vs. full checkpoints. Smoke-tested on tiny
  val checkpoints; real use should point it at trained `best_checkpoint.pt` files.
- **2026-08-11 (Codex):** Added
  `notebooks/disagreement_aware_probunet_colab.ipynb` as the single-notebook Colab
  entry point with all project code included directly in notebook cells. It
  defaults to `RUN_MODE = "smoke"` so Run All completes quickly; change to
  `"final"` for 240k-step ablation runs.
- **2026-08-12 (Claude):** First full 240k ablation runs finished
  (`outputs/lidc_ablation/{baseline,head,full}`) and **all three suffer posterior
  collapse — do not report these numbers.** `loss_kl` decays to exactly 0 by
  ~30k steps, the prior scale sits at 1.0, and measured sample diversity
  `E[d(S,S')]` is 0.006 at 240k vs 0.45 at step 1k. Each run's
  `best_checkpoint.pt` is from step 1000–2000, which is the tell. Val GED
  therefore *rises* 0.34 → 0.61 while dice improves — the net became
  deterministic. Likely cause: `train_lidc_ablation.py` uses `nn.NLLLoss()`
  (reduction `mean`, so per-pixel), while Kohl et al. sum the reconstruction CE
  over pixels; at 128×128 that under-weights reconstruction ~16k× relative to
  `beta * KL`, so β = 1 behaves like β ≈ 16000. Fix the reduction (or rescale β)
  before rerunning. Second issue: `evaluate()` calls `dice`/`jaccard` with
  `nan_for_nonexisting=False`, so a correct empty-vs-empty prediction scores 0
  instead of being excluded — this deflates `val_dice` on a dataset where 65% of
  crops have an empty grader mask. Appendix B already handles the empty case for
  GED; the dice path should too.
