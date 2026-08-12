import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from probunet.disagreement import compute_disagreement, model_uncertainty_from_samples
from probunet.lidc import LIDCCrops
from train_lidc_ablation import make_model


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_args(checkpoint, device, variant):
    saved = dict(checkpoint.get("args") or {})
    saved["device"] = device
    saved["variant"] = variant
    return SimpleNamespace(**saved)


def load_model(checkpoint_path, variant, device):
    checkpoint = torch_load(checkpoint_path, device)
    args = checkpoint_args(checkpoint, device, variant)
    model = make_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def sample_model(model, image, n_samples, has_disagreement_head):
    with torch.no_grad():
        model.reset()
        outputs = model.sample_prior(n_samples, out_device=image.device, input_=image)
        outputs = torch.stack(outputs, dim=0)
        masks = torch.argmax(outputs, dim=2).cpu().numpy().astype(np.float32)
        uncertainty = model_uncertainty_from_samples(outputs).cpu().numpy()
        predicted_disagreement = None
        if has_disagreement_head:
            predicted_disagreement = model.predict_disagreement().cpu().numpy()
    return masks[:, 0], uncertainty[0, 0], (
        None if predicted_disagreement is None else predicted_disagreement[0, 0]
    )


def disagreement_score(dataset, index):
    sample = dataset[index]
    disagreement = compute_disagreement(sample["masks"][None]).numpy()[0, 0]
    return float(disagreement.mean())


def choose_indices(dataset, requested_indices, num_cases, selection):
    if requested_indices:
        return requested_indices[:num_cases]
    if selection == "first":
        return list(range(min(num_cases, len(dataset))))

    scores = []
    for index in tqdm(range(len(dataset)), desc="rank disagreement", dynamic_ncols=True):
        scores.append((disagreement_score(dataset, index), index))
    scores.sort(reverse=True)

    # LIDC crops are consecutive slices through the same nodule, so the top of this
    # ranking is one high-disagreement lesion repeated dozens of times -- without
    # this, "8 cases" is really one case shown 8 times. Take one crop per patient
    # first, and only reuse patients if that does not fill num_cases.
    chosen, seen, leftovers = [], set(), []
    for _, index in scores:
        patient = os.path.basename(os.path.dirname(dataset.samples[index][0]))
        if patient in seen:
            leftovers.append(index)
            continue
        seen.add(patient)
        chosen.append(index)
        if len(chosen) == num_cases:
            return chosen
    return (chosen + leftovers)[:num_cases]


def add_panel(ax, image, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def make_case_figure(
    out_path,
    image,
    masks,
    human_disagreement,
    baseline_samples,
    baseline_uncertainty,
    full_samples,
    full_uncertainty,
    predicted_disagreement,
):
    n_samples = baseline_samples.shape[0]
    ncols = max(6, n_samples + 2)
    fig, axes = plt.subplots(4, ncols, figsize=(2.1 * ncols, 8.2))

    for ax in axes.ravel():
        ax.axis("off")

    add_panel(axes[0, 0], image, "input")
    for grader in range(masks.shape[0]):
        add_panel(axes[0, grader + 1], masks[grader], "human M{}".format(grader + 1))
    add_panel(axes[0, 5], human_disagreement, "human D", cmap="magma", vmin=0.0, vmax=1.0)

    axes[1, 0].set_title("baseline samples", fontsize=8)
    axes[1, 0].axis("off")
    for i in range(n_samples):
        add_panel(axes[1, i + 1], baseline_samples[i], "S{}".format(i + 1))
    add_panel(
        axes[1, n_samples + 1],
        baseline_uncertainty,
        "baseline U",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    axes[2, 0].set_title("full samples", fontsize=8)
    axes[2, 0].axis("off")
    for i in range(n_samples):
        add_panel(axes[2, i + 1], full_samples[i], "S{}".format(i + 1))
    add_panel(
        axes[2, n_samples + 1],
        full_uncertainty,
        "full U",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    add_panel(
        axes[3, 0],
        predicted_disagreement,
        "predicted D",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    add_panel(axes[3, 1], np.abs(full_uncertainty - human_disagreement), "|U-D|", cmap="magma")
    add_panel(
        axes[3, 2],
        np.abs(predicted_disagreement - human_disagreement),
        "|Dhat-D|",
        cmap="magma",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--data-root", default="data/lidc")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out-dir", default="outputs/lidc_figures")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--selection", choices=("highest-disagreement", "first"), default="highest-disagreement")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    baseline, baseline_checkpoint = load_model(args.baseline_checkpoint, "baseline", args.device)
    full, full_checkpoint = load_model(args.full_checkpoint, "full", args.device)

    baseline_crop = (baseline_checkpoint.get("args") or {}).get("crop_size", 128)
    full_crop = (full_checkpoint.get("args") or {}).get("crop_size", baseline_crop)
    if baseline_crop != full_crop:
        raise ValueError(
            "baseline and full checkpoints use different crop sizes: {} vs {}".format(
                baseline_crop,
                full_crop,
            )
        )

    dataset = LIDCCrops(
        root=args.data_root,
        split=args.split,
        crop_size=baseline_crop,
        train=False,
        single_random_grader=False,
    )
    indices = choose_indices(dataset, args.indices, args.num_cases, args.selection)

    for index in tqdm(indices, desc="figures", dynamic_ncols=True):
        sample = dataset[index]
        image = sample["image"][None].to(args.device)
        masks = sample["masks"].numpy()
        human_disagreement = compute_disagreement(sample["masks"][None]).numpy()[0, 0]

        baseline_samples, baseline_uncertainty, _ = sample_model(
            baseline,
            image,
            args.samples,
            has_disagreement_head=False,
        )
        full_samples, full_uncertainty, predicted_disagreement = sample_model(
            full,
            image,
            args.samples,
            has_disagreement_head=True,
        )

        out_path = os.path.join(args.out_dir, "case_{:04d}.png".format(index))
        make_case_figure(
            out_path,
            sample["image"][0].numpy(),
            masks,
            human_disagreement,
            baseline_samples,
            baseline_uncertainty,
            full_samples,
            full_uncertainty,
            predicted_disagreement,
        )
        print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
