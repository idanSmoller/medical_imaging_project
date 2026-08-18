"""Final test-set evaluation of trained LIDC ablation checkpoints (plan Phase 9).

Evaluates one or more ``best_checkpoint.pt`` files on the held-out test split and
reports, per variant:

* segmentation quality (plan section 14): dice, IoU
* distribution quality (plan section 15): generalized energy distance
* disagreement alignment (plan section 16): MAE and correlation between the model
  uncertainty map U and the human disagreement map D_GT, computed **two ways** --
  the plan's section 9 entropy-of-the-mean, and mutual information, which unlike
  the former is zero when the samples are identical
* diversity diagnostics: E[d(S,S')] and the fraction of images whose samples are
  strictly nested. Nested samples mean the latent only rescales one mask, which
  caps what any diversity-shaping loss can do.
* the disagreement head (plan section 22 Q1), against a **trivial control**:
  disagreement in LIDC sits largely on lesion boundaries, so a predictor that just
  outlines the model's own segmentation already scores well. The head has to beat
  that number to mean anything.

Metric implementations are imported, never reimplemented (see AGENTS.md).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_lidc_ablation import make_model, sample_masks

from probunet.disagreement import (
    compute_disagreement,
    model_uncertainty_from_samples,
    sample_diversity_from_samples,
)
from probunet.eval import (
    binary_iou_distance,
    dice,
    disagreement_correlation,
    disagreement_mae,
    generalized_energy_distance,
    jaccard,
)
from probunet.lidc import LIDCCrops


VARIANTS = ("baseline", "head", "full")


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved = dict(checkpoint["args"])
    saved["device"] = device
    model = make_model(argparse.Namespace(**saved))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, saved["variant"]


def boundary_band(mask, kernel=5):
    """Morphological gradient of a binary mask: dilation minus erosion.

    Implemented with max-pooling because scipy is not installed in this env.
    ``mask`` is a float tensor shaped (B, 1, H, W).
    """

    dilated = F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)
    return dilated - eroded


def nested_and_diversity(sample_masks_np):
    """Return (mean pairwise 1-IoU, whether all samples are nested) for one image.

    ``sample_masks_np`` is a boolean array shaped (K, H, W).
    """

    k = sample_masks_np.shape[0]
    distances = [
        binary_iou_distance(sample_masks_np[i], sample_masks_np[j])
        for i in range(k)
        for j in range(k)
        if i != j
    ]
    order = np.argsort(sample_masks_np.reshape(k, -1).sum(axis=1))
    nested = all(
        not np.logical_and(sample_masks_np[order[i]], ~sample_masks_np[order[i + 1]]).any()
        for i in range(k - 1)
    )
    return float(np.mean(distances)) if distances else 0.0, bool(nested)


def evaluate_checkpoint(model, variant, loader, args):
    """Test-set metrics for one checkpoint. Mirrors train_lidc_ablation.evaluate,
    with the diversity diagnostics and the trivial control added."""

    acc = {key: [] for key in (
        "dice", "iou", "ged",
        "unc_mae", "unc_corr", "div_mae", "div_corr",
        "pairwise_distance", "nested",
        "pred_mae", "pred_corr", "control_corr",
    )}

    with torch.no_grad():
        for batch in tqdm(loader, desc=variant, leave=False, dynamic_ncols=True):
            image = batch["image"].to(args.device)
            masks = batch["masks"].to(args.device)

            model.reset()
            prediction = model(image)
            pred_mask = torch.argmax(prediction, dim=1).cpu().numpy()

            outputs, prior_masks = sample_masks(model, image, args.eval_samples)
            uncertainty = model_uncertainty_from_samples(outputs).cpu().numpy()
            diversity = sample_diversity_from_samples(outputs).cpu().numpy()
            human = compute_disagreement(masks).cpu().numpy()

            # Trivial Q1 control: outline the model's own deterministic prediction.
            deterministic = torch.argmax(prediction, dim=1, keepdim=True).float()
            control = boundary_band(deterministic).cpu().numpy()

            predicted = None
            if variant in ("head", "full", "distribution"):
                predicted = model.predict_disagreement().cpu().numpy()

            grader_masks = masks.cpu().numpy().astype(bool)
            prior_masks = prior_masks.cpu().numpy().astype(bool)

            for i in range(image.shape[0]):
                for grader in range(grader_masks.shape[1]):
                    acc["dice"].append(
                        dice(pred_mask[i] != 0, grader_masks[i, grader], nan_for_nonexisting=True)
                    )
                    acc["iou"].append(
                        jaccard(pred_mask[i] != 0, grader_masks[i, grader], nan_for_nonexisting=True)
                    )
                acc["ged"].append(generalized_energy_distance(prior_masks[:, i], grader_masks[i]))

                acc["unc_mae"].append(disagreement_mae(uncertainty[i, 0], human[i, 0]))
                acc["unc_corr"].append(disagreement_correlation(uncertainty[i, 0], human[i, 0]))
                acc["div_mae"].append(disagreement_mae(diversity[i, 0], human[i, 0]))
                acc["div_corr"].append(disagreement_correlation(diversity[i, 0], human[i, 0]))
                acc["control_corr"].append(disagreement_correlation(control[i, 0], human[i, 0]))

                distance, nested = nested_and_diversity(prior_masks[:, i])
                acc["pairwise_distance"].append(distance)
                acc["nested"].append(float(nested))

                if predicted is not None:
                    acc["pred_mae"].append(disagreement_mae(predicted[i, 0], human[i, 0]))
                    acc["pred_corr"].append(disagreement_correlation(predicted[i, 0], human[i, 0]))

    return {
        key: (float(np.nanmean(values)) if values else float("nan"))
        for key, values in acc.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default="outputs_12_08_26_240000/lidc_ablation",
        help="Directory holding <variant>/best_checkpoint.pt subdirectories.",
    )
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--checkpoint-name", default="best_checkpoint.pt")
    parser.add_argument("--data-root", default="data/lidc")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.runs_dir.rstrip("/")), "final_eval")
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = LIDCCrops(
        root=args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        train=False,
    )
    if args.max_test is not None and args.max_test < len(dataset):
        dataset = Subset(dataset, range(args.max_test))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    print("evaluating on {} split: {} images, {} samples each".format(
        args.split, len(dataset), args.eval_samples))

    results = []
    for variant in args.variants:
        path = os.path.join(args.runs_dir, variant, args.checkpoint_name)
        if not os.path.exists(path):
            print("skipping {}: no checkpoint at {}".format(variant, path))
            continue
        model, checkpoint, saved_variant = load_checkpoint_model(path, args.device)
        metrics = evaluate_checkpoint(model, saved_variant, loader, args)
        metrics["variant"] = variant if saved_variant == "distribution" else saved_variant
        metrics["step"] = int(checkpoint.get("step", -1))
        metrics["lambda_disagreement"] = checkpoint["args"].get("lambda_disagreement")
        metrics["lambda_alignment"] = checkpoint["args"].get("lambda_alignment")
        results.append(metrics)
        print("{}: step {} dice {:.4f} ged {:.4f} unc_corr {:.4f} div_corr {:.4f} "
              "nested {:.0%}".format(
                  saved_variant, metrics["step"], metrics["dice"], metrics["ged"],
                  metrics["unc_corr"], metrics["div_corr"], metrics["nested"]))

    if not results:
        raise SystemExit("no checkpoints evaluated")

    csv_path = os.path.join(args.out_dir, "test_metrics.csv")
    fieldnames = ["variant", "step", "lambda_disagreement", "lambda_alignment"] + [
        k for k in results[0] if k not in
        ("variant", "step", "lambda_disagreement", "lambda_alignment")
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    control = float(np.nanmean([r["control_corr"] for r in results]))
    lines = [
        "# LIDC {} results ({} samples per image)".format(args.split, args.eval_samples),
        "",
        "Checkpoints from `{}`.".format(args.runs_dir),
        "",
        "| variant | step | dice | IoU | GED | U-D corr (entropy) | U-D corr (mutual info) "
        "| pred D corr | E[d(S,S')] | nested |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        pred = "n/a" if np.isnan(r["pred_corr"]) else "{:.4f}".format(r["pred_corr"])
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {:.4f} | {:.0%} |".format(
                r["variant"], r["step"], r["dice"], r["iou"], r["ged"],
                r["unc_corr"], r["div_corr"], pred, r["pairwise_distance"], r["nested"])
        )
    lines += [
        "",
        "**Trivial control for the disagreement head (plan section 22 Q1):** the boundary "
        "band of the model's own predicted segmentation correlates with D_GT at "
        "**{:.4f}**. The head's `pred D corr` must beat this to show it learned "
        "something about raters rather than about lesion outlines.".format(control),
        "",
        "`nested` is the fraction of images whose samples are all nested inside one "
        "another. A high value means the latent only rescales a single mask, which "
        "bounds what any diversity-shaping loss can achieve.",
    ]
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    with open(os.path.join(args.out_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    print("\ntrivial control (boundary band of own prediction) vs D_GT: {:.4f}".format(control))
    print("wrote {} and {}".format(csv_path, md_path))


if __name__ == "__main__":
    main()
