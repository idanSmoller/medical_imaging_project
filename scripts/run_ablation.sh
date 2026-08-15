#!/bin/bash
# Launch one LIDC ablation variant.
#
#   bash scripts/run_ablation.sh <gpu> <baseline|head|full> [extra args...]
#
# Defaults encode what the 2026-08-15 runs used. Two of them are load-bearing and
# should not be raised without rereading the handoff log in AGENTS.md:
#
#   --lambda-disagreement 0.01   L_D is summed over pixels, so 0.5 makes the
#                                auxiliary term ~3x the reconstruction.
#   --lambda-alignment    1e-5   1e-4 diverges within ~2k steps: the alignment
#                                loss backprops into the prior and inflating the
#                                prior variance is the cheapest way to raise
#                                sample diversity, so KL doubles every step.
#
# Run it inside tmux -- a 100k run takes ~3.7h (baseline/head) or ~6h (full).
set -u

GPU=${1:?usage: run_ablation.sh <gpu> <variant> [extra args...]}
VARIANT=${2:?usage: run_ablation.sh <gpu> <variant> [extra args...]}
shift 2

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="$GPU"
echo "launching $VARIANT on GPU $GPU at $(date)"

python scripts/train_lidc_ablation.py \
  --variant "$VARIANT" --steps 100000 \
  --lambda-disagreement 0.01 --lambda-alignment 1e-5 \
  --out-dir "outputs_fcombfix/lidc_ablation/$VARIANT" "$@"

echo "=== $VARIANT DONE rc=$? at $(date) ==="
