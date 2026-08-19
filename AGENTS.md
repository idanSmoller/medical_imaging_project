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

**Read `RESULTS.md` for the experimental record** — all numbers, the three arms,
the open questions, and how to get the checkpoints. Summary of where things are:

Done: LIDC download + loader, disagreement utilities, disagreement head and
losses, the full Phase 8 ablation at 240k steps, Phase 9 test-set evaluation
(`scripts/eval_lidc_final.py`), and Phase 10 qualitative figures.

Both ablation arms are complete: affine at 240k and fcomb-fixed at 100k, each
with a test-set table. Not done: the two arms do not share an LR schedule, so
their absolute GEDs are not comparable — **rerunning the affine arm at 100k is
the top open task**. The plan's λ sweep is deliberately skipped; see RESULTS.md §6.

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
- **2026-08-15 (Claude):** Phase 9/10 done for the 240k arm, and a fcomb-fixed
  rerun launched. **Test-set results** (`outputs_12_08_26_240000/final_eval/`,
  1980 images, 16 samples): baseline dice 0.327 / GED 0.316, head 0.328 / 0.314,
  full 0.289 / 0.330. **Q1 is positive:** the head's `predD_corr` is 0.617 against
  an *image-only* trivial control of 0.545 (boundary band of the model's own
  prediction). An earlier 0.654 figure was the boundary band of the GT majority
  mask — an oracle, not a fair control; do not use it. **Q2 is marginal:**
  `unc_corr` 0.584 / 0.597 / 0.608, i.e. +0.024 for full over baseline at a 12%
  dice cost. The mutual-info metric tracks entropy-of-mean within 0.01, so the
  "blind metric" concern is minor in practice — **nestedness was the real
  constraint: 100% of test images had strictly nested samples in all three
  variants.**
  New: `scripts/eval_lidc_final.py` (test eval + diversity diagnostics + the
  trivial control), figure selection now dedups by patient.
  **Two things bit hard when the fcomb nonlinearity was enabled, both now fixed:**
  (a) all three variants NaN in the prior encoder within ~2k steps — this branch's
  `nn.NLLLoss(reduction="sum")` sums over batch *and* pixels, so gradients are
  ~32x the paper's per-image convention. Added `--grad-clip` (default 100).
  (b) `full` additionally diverges at `lambda_alignment=1e-4`: KL doubles every
  step to overflow, because the alignment loss backprops into the prior through
  `sample_prior_train` and inflating the prior variance is the cheapest way to
  raise diversity. Probed 1e-5 stable (KL settles ~43); **the rerun uses 1e-5, not
  the 1e-4 of the 240k arm** — note this when comparing the two arms.
  Running now in tmux probunet/probunet2/probunet3 on GPU 2/3/5, 100k steps, into
  `outputs_fcombfix/lidc_ablation/`. When they finish, rerun
  `scripts/eval_lidc_final.py --runs-dir outputs_fcombfix/lidc_ablation` and
  compare `nested` and `div_corr` against the 240k arm — that is the real test of
  plan §22 Q2.
- **2026-08-15 (Claude), results handoff:** Added **`RESULTS.md`** — the
  experimental record (all three arms, the test table, the three defects behind
  the Q2 null, stability gotchas, and how to obtain checkpoints). Read it before
  touching experiments. Also added `scripts/eval_lidc_final.py` (test eval +
  diversity/nestedness diagnostics + the image-only Q1 control) and
  `scripts/run_ablation.sh`. Three code fixes landed with the fcomb change:
  `--grad-clip` (default 100), a prior/posterior `logvar` clamp at ±10
  (`LOGVAR_MIN/MAX` in `model.py`), and the `reduce-{i}-nonlin` call in
  `InjectionUNet.forward`. **`outputs_fcombfix/lidc_ablation/full/` was mid-run at
  commit time** — its `history.csv` here is a partial snapshot; rerun
  `scripts/eval_lidc_final.py --runs-dir outputs_fcombfix/lidc_ablation` once it
  finishes. Checkpoints are gitignored (~5.9 GB); RESULTS.md §8 lists transfer
  options, but retraining from `args.json` is reproducible and usually easier.
- **2026-08-16 (Claude):** fcomb-fixed arm complete (all three variants, 100k).
  **`full` is now the best variant** — best GED (0.3884), highest `unc_corr`
  (0.5086) and `predD_corr` (0.6270), at only 4% dice below baseline, versus 12%
  in the affine arm. **Nestedness dropped 100% → 1–3%.** So plan §22 Q2 flips from
  null to weakly positive once the latent can vary spatially, and Q1 strengthens
  to +0.18 over the trivial control. RESULTS.md §4 has the table.
  **Report `outputs_fcombfix/final_eval_100k/` (matched 100k `latest_checkpoint`),
  not `outputs_fcombfix/final_eval/`** — best-GED selection picks step 1000
  (baseline) / 3000 (head) in this arm, i.e. barely-trained models, against step
  71000 for full; it compares training length, not methods.
  Cross-arm GED (0.388 vs 0.316) is still confounded by the LR schedule; running
  the affine arm at 100k remains the top open task.
- **2026-08-17 (Codex):** Created
  `plans/multi_rater_distribution_matching_design.md` to capture the new theory
  pivot: model the four LIDC masks as an empirical/smoothed distribution over
  masks, train model samples with a two-way soft Dice/IoU kernel likelihood
  (sections 8.2/8.3), and treat disagreement as the entropy/readout of that
  distribution rather than only an auxiliary correction. This is a design note,
  not implemented code yet.
- **2026-08-17 (Codex):** Implemented the minimal distribution-matching training
  variant on branch `multi-rater-distribution-matching`. New
  `--variant distribution` samples from the prior with gradients and trains with
  `lambda_distribution * L_dist + lambda_consensus * L_consensus +
  lambda_disagreement * L_D`; it deliberately skips random-grader CE, posterior
  KL, and old alignment. `--distribution-mode coverage` implements plan §8.2 and
  `--distribution-mode kernel` implements §8.3, both with soft Dice distances.
  CPU smoke-tested both modes for one step under `outputs/distribution_smoke/`.
- **2026-08-18 (Claude):** Distribution-matching arm finished (both modes, 100k,
  GPUs 2/3) and evaluated on test. **It is a negative result: best dice of any
  arm (0.4383 coverage), worst GED of any arm (0.7135 / 0.9329), and
  `E[d(S,S')]` = 0.024 / 0.0004 with 99–100% nested — the samples collapse to a
  single mask.** Numbers and the mechanism are in RESULTS.md §4b; do not "fix" it
  with a different τ, it needs an explicit diversity term or the KL back.
  `scripts/eval_lidc_final.py` gained two lines for this: `"distribution"` added
  to the disagreement-head gate (it was silently reporting `pred_corr = n/a`), and
  rows now label by run directory so `coverage`/`kernel` don't both print as
  `distribution`. Also generated `outputs_fcombfix/figures_100k/` (qualitative
  grids from the *reported* 100k checkpoints; the ones in
  `outputs_12_08_26_240000/figures/` are from the affine arm).
  **Report decision: the fcomb-fixed arm stays the headline extension and the
  distribution arm is reported as a fourth arm / negative result.** The full
  6-page report now lives in `overleaf/` (`main.tex` + `figure_model.tex` +
  `figs/`, compiles clean with pdflatex at 10pt). Open item before submission:
  `notebooks/disagreement_aware_probunet_colab.ipynb` predates the distribution
  variant, so it does not contain arm D — either port it or scope the notebook's
  claim in the report.
  Also added **`scripts/make_figure_assets.py`**, which `overleaf/figure_model.tex`
  had referenced for months without it existing. It renders every thumbnail of the
  architecture figure from one test crop (`LIDC-IDRI-0217/z-129.0_c0`) and the
  reported `full` checkpoint. Before this, `U`, `D_hat` and the predicted
  segmentation were hand-drawn rings/blobs sitting next to a *real* `D_GT`, which
  implied a match the model does not achieve; and the `Q^(k)` thumbnails were
  argmax masks although `U` is the entropy of the mean **softmax probability**.
  Rerun it if the reported checkpoint changes.
- **2026-08-18 (Claude), checkpoint/code trap:** **The `outputs_11_08_26/` and
  `outputs_12_08_26_*/` checkpoints can no longer be evaluated with current
  `probunet/model.py`.** The fcomb fix (598e1df) added `reduce-{i}-nonlin`
  modules to `InjectionUNet.forward`; LeakyReLU has zero parameters, so
  `load_state_dict` succeeds *silently* while the forward pass now applies
  activations that were absent during training. Re-evaluating the 240k affine arm
  this way gave `E[d(S,S')]` 0.356 (step 2k) / 0.233 (step 240k) and GED that
  *falls* with training, contradicting that run's own `history.csv` — those
  numbers are artefacts, not results. To re-measure a pre-fix arm you must check
  out the pre-598e1df code. Consequence for the report: the handoff-log claim
  "sample diversity fell from 0.45 to 0.006" is **not reproducible** and was
  removed from `overleaf/main.tex`; the surviving posterior-collapse claims (KL
  reaches zero by step 31k, val GED rises 0.34 -> 0.61) come from that run's own
  `history.csv` and are sound.
  Also: `overleaf/main.tex` is now the **prose** report (6 pages, compiles clean);
  the bullet-point version is kept as `overleaf/main_bullets.tex.bak`.
- **2026-08-19 (Claude), report scope change:** The submitted report now covers
  **three arms only (A/B/C)** — the collaborator removed the distribution-matching
  arm D from `overleaf/main.tex`. Arm D's runs, numbers and analysis are unchanged
  and remain in RESULTS.md §4b; only the write-up dropped it. `overleaf/main.tex`
  is prose, 6 pages, and was rewritten against ~18 inline `% Q:` review comments.
  Two edits made while removing arm D had broken the build and are fixed: a
  literal TAB had replaced the `\t` of `\text` in `$L_{\text{align}}$`, and
  deleting the `gao2023` entry also deleted `\end{thebibliography}}`.
  **When editing `overleaf/main.tex`, hand back targeted edits, not whole-file
  rewrites** — the collaborator edits the same file in Overleaf, and Overleaf
  comments are anchored to text ranges, so a wholesale replacement orphans them.
