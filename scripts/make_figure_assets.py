"""Render the thumbnails used by overleaf/figure_model.tex from real data.

Every panel of the architecture figure comes from one LIDC test crop and one
trained checkpoint, so the CT image, the four grader masks, D_GT, the model's
samples, U and D_hat all describe the *same* case and are directly comparable.

    python scripts/make_figure_assets.py \
        --checkpoint outputs_fcombfix/lidc_ablation/full/latest_checkpoint.pt

D_GT, U and D_hat are written on a shared 0..1 colour scale on purpose. They are
meant to be read against each other in the figure, so per-panel normalisation
would invent an agreement that is not there.

Every panel is zoomed to the same lesion-centred window (--zoom-margin). At the
6 mm the figure gives each thumbnail, a nodule inside the full 128x128 crop is
a few pixels across and reads as an empty black square.
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lidc_ablation import make_model

from probunet.disagreement import compute_disagreement, model_uncertainty_from_samples
from probunet.lidc import LIDCCrops

SIZE_PX = 296
CMAP = "inferno"


def lesion_window(masks, margin=1.8, min_size=24):
    """Square window centred on the union of the grader masks.

    Returns (top, left, size) in pixels, clipped to the image.
    """

    union = masks.max(axis=0) > 0.5
    height, width = union.shape
    if not union.any():
        return 0, 0, min(height, width)
    rows, cols = np.where(union)
    centre_y = 0.5 * (rows.min() + rows.max())
    centre_x = 0.5 * (cols.min() + cols.max())
    extent = max(rows.max() - rows.min(), cols.max() - cols.min()) + 1
    size = int(max(min_size, round(extent * margin)))
    size = min(size, height, width)
    top = int(round(centre_y - size / 2.0))
    left = int(round(centre_x - size / 2.0))
    top = max(0, min(top, height - size))
    left = max(0, min(left, width - size))
    return top, left, size


def save_map(array, path, window=None, cmap=CMAP, vmin=0.0, vmax=1.0):
    """Write one square thumbnail with no axes, padding or interpolation."""

    if window is not None:
        top, left, size = window
        array = array[top:top + size, left:left + size]
    fig = plt.figure(figsize=(SIZE_PX / 100.0, SIZE_PX / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(path, dpi=100, facecolor="black")
    plt.close(fig)


def pick_distinct_samples(sample_masks, count=3, min_area_frac=0.15):
    """Indices of `count` draws chosen greedily for mutual difference.

    Empty draws and slivers below `min_area_frac` of the largest draw are skipped:
    at thumbnail size they render as blank squares and read as a broken figure
    rather than as a sample. The caption says how many draws were empty, so the
    filtering is stated rather than hidden.
    """

    areas = sample_masks.reshape(len(sample_masks), -1).sum(axis=1)
    floor = min_area_frac * areas.max() if areas.max() > 0 else 0
    candidates = [k for k in range(len(sample_masks)) if areas[k] >= max(1, floor)]
    if len(candidates) <= count:
        return candidates

    def distance(a, b):
        union = np.logical_or(sample_masks[a] > 0, sample_masks[b] > 0).sum()
        inter = np.logical_and(sample_masks[a] > 0, sample_masks[b] > 0).sum()
        return 1.0 - inter / union if union else 0.0

    chosen = [max(candidates, key=lambda k: areas[k])]
    while len(chosen) < count:
        chosen.append(max(
            (k for k in candidates if k not in chosen),
            key=lambda k: min(distance(k, c) for c in chosen)))
    return chosen


def find_index(dataset, patient, stem):
    for index, (image_path, _) in enumerate(dataset.samples):
        if patient in image_path and os.path.basename(image_path) == stem + ".png":
            return index
    raise SystemExit("case {}/{} not found in split".format(patient, stem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="outputs_fcombfix/lidc_ablation/full/latest_checkpoint.pt")
    parser.add_argument("--data-root", default="data/lidc")
    parser.add_argument("--split", default="test")
    parser.add_argument("--patient", default="LIDC-IDRI-0217")
    parser.add_argument("--stem", default="z-129.0_c0")
    parser.add_argument("--out-dir", default="overleaf/figs")
    parser.add_argument("--samples", type=int, default=16,
                        help="Prior samples behind U; 3 of them are also written out.")
    parser.add_argument("--zoom-margin", type=float, default=1.8,
                        help="Window size as a multiple of the lesion bounding box.")
    parser.add_argument("--min-area-frac", type=float, default=0.15,
                        help="Skip sample draws smaller than this fraction of the largest.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = LIDCCrops(root=args.data_root, split=args.split, crop_size=128, train=False)
    sample = dataset[find_index(dataset, args.patient, args.stem)]
    image = sample["image"][None].to(args.device)
    masks = sample["masks"][None].to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    saved = dict(checkpoint["args"])
    saved["device"] = args.device
    model = make_model(argparse.Namespace(**saved))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        model.reset()
        prediction = model(image)
        predicted_mask = torch.argmax(prediction, dim=1)[0].cpu().numpy()
        outputs = torch.stack(
            model.sample_prior(args.samples, out_device=image.device, input_=image), dim=0)
        uncertainty = model_uncertainty_from_samples(outputs)[0, 0].cpu().numpy()
        predicted_disagreement = model.predict_disagreement()[0, 0].cpu().numpy()
        sample_masks = torch.argmax(outputs, dim=2)[:, 0].cpu().numpy()
        # The figure labels these Q^(k), and U is the entropy of the MEAN
        # FOREGROUND PROBABILITY over samples (disagreement.model_uncertainty_
        # from_samples), not of thresholded masks. So show the probability maps
        # the math actually refers to; argmax masks are used only to rank the
        # draws by how different they are.
        sample_probs = torch.softmax(outputs, dim=2)[:, 0, 1].cpu().numpy()

    human = compute_disagreement(masks)[0, 0].cpu().numpy()
    grader_masks = sample["masks"].numpy()
    window = lesion_window(grader_masks, margin=args.zoom_margin)

    save_map(sample["image"][0].numpy(), os.path.join(args.out_dir, "ct.png"),
             window, cmap="gray")
    for grader in range(masks.shape[1]):
        save_map(grader_masks[grader],
                 os.path.join(args.out_dir, "mask{}.png".format(grader)), window, cmap="gray")
    save_map(human, os.path.join(args.out_dir, "dgt.png"), window)
    save_map(uncertainty, os.path.join(args.out_dir, "u.png"), window)
    save_map(predicted_disagreement, os.path.join(args.out_dir, "dhat.png"), window)
    save_map(predicted_mask, os.path.join(args.out_dir, "pred.png"), window, cmap="gray")

    # The figure shows three sample thumbnails and its whole point is that they
    # differ, so pick the three most mutually distinct non-empty draws rather
    # than the first three -- consecutive draws are often identical.
    chosen = pick_distinct_samples(sample_masks, count=3,
                                   min_area_frac=args.min_area_frac)
    for slot, k in enumerate(chosen):
        save_map(sample_probs[k], os.path.join(args.out_dir, "sample{}.png".format(slot)),
                 window, cmap="gray")

    empty = [g for g in range(masks.shape[1]) if sample["masks"][g].sum() == 0]
    print("case {}/{}".format(args.patient, args.stem))
    print("  empty grader masks: {}".format(empty or "none"))
    print("  D_GT   max {:.3f}  mean-over-nonzero {:.3f}".format(
        human.max(), human[human > 0].mean() if (human > 0).any() else 0.0))
    print("  U      max {:.3f}  mean-over-nonzero {:.3f}".format(
        uncertainty.max(), uncertainty[uncertainty > 1e-3].mean()
        if (uncertainty > 1e-3).any() else 0.0))
    print("  D_hat  max {:.3f}".format(predicted_disagreement.max()))
    print("  distinct sample masks: {} of {}; empty: {}".format(
        len(np.unique(sample_masks.reshape(args.samples, -1), axis=0)), args.samples,
        int((sample_masks.reshape(args.samples, -1).sum(axis=1) == 0).sum())))
    print("  window (top, left, size): {}".format(window))
    print("  sample thumbnails from draws {}".format(chosen))
    print("wrote {}".format(args.out_dir))


if __name__ == "__main__":
    main()
