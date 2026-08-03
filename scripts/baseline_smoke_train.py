import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from probunet.eval import dice
from probunet.model import ProbabilisticSegmentationNet, InjectionConvEncoder, InjectionUNet


def boxes_to_masks(boxes, height, width):
    masks = np.zeros((boxes.shape[0], height, width), dtype=np.int64)
    for i, seq_boxes in enumerate(boxes):
        for x1, y1, x2, y2 in seq_boxes:
            x1 = int(np.clip(round(x1), 0, width - 1))
            x2 = int(np.clip(round(x2), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            y2 = int(np.clip(round(y2), 0, height - 1))
            if x2 > x1 and y2 > y1:
                masks[i, y1:y2 + 1, x1:x2 + 1] = 1
    return masks


def load_box_segmentation_data(npz_path, split, max_items):
    with np.load(npz_path) as data:
        images = data[f"{split}_images"][:max_items].astype(np.float32) / 255.0
        boxes = data[f"{split}_boxes"][:max_items]

    masks = boxes_to_masks(boxes, images.shape[-2], images.shape[-1])
    images = images[:, :1]
    return torch.from_numpy(images), torch.from_numpy(masks[:, None])


def make_model(device):
    return ProbabilisticSegmentationNet(
        in_channels=1,
        out_channels=2,
        num_feature_maps=8,
        latent_size=2,
        depth=3,
        latent_distribution=distributions.Normal,
        task_op=InjectionUNet,
        task_kwargs={
            "injection_channels": 2,
            "output_activation_op": nn.LogSoftmax,
            "output_activation_kwargs": {"dim": 1},
            "activation_kwargs": {"inplace": True},
        },
        prior_op=InjectionConvEncoder,
        prior_kwargs={
            "in_channels": 1,
            "out_channels": 4,
            "depth": 3,
            "num_feature_maps": 8,
            "activation_kwargs": {"inplace": True},
            "norm_depth": 0,
        },
        posterior_op=InjectionConvEncoder,
        posterior_kwargs={
            "in_channels": 3,
            "out_channels": 4,
            "depth": 3,
            "num_feature_maps": 8,
            "activation_kwargs": {"inplace": True},
            "norm_depth": 0,
        },
    ).to(device)


def save_qualitative(out_dir, image, target, prediction, samples):
    fig, axes = plt.subplots(1, 7, figsize=(14, 2))
    panels = [image, target, prediction] + samples
    titles = ["image", "target", "mean"] + [f"sample {i + 1}" for i in range(len(samples))]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qualitative_samples.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="/home/idan.smoller/sandbox/scam/HW4/hw4_prepared_lesion_sequences.npz")
    parser.add_argument("--out-dir", default="outputs/baseline_smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-val", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1)
    np.random.seed(1)

    train_x, train_y = load_box_segmentation_data(args.npz, "source_train", args.max_train)
    val_x, val_y = load_box_segmentation_data(args.npz, "source_val", args.max_val)
    train_x, train_y = train_x.to(args.device), train_y.to(args.device)
    val_x, val_y = val_x.to(args.device), val_y.to(args.device)

    model = make_model(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()

    history = []
    for epoch in range(args.epochs):
        permutation = torch.randperm(train_x.shape[0], device=args.device)
        losses = []
        for start in range(0, train_x.shape[0], args.batch_size):
            idx = permutation[start:start + args.batch_size]
            x = train_x[idx]
            y = train_y[idx]

            model.reset()
            model.train()
            optimizer.zero_grad()
            prediction = model(x, y, make_onehot=True, make_onehot_classes=(0, 1))
            loss_seg = criterion(prediction, y[:, 0].long())
            loss_kl = distributions.kl_divergence(model.posterior, model.prior).mean()
            loss = loss_seg + 1e-3 * loss_kl
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.reset()
        model.eval()
        with torch.no_grad():
            prediction = model(val_x, val_y, make_onehot=True, make_onehot_classes=(0, 1))
            pred_mask = torch.argmax(prediction, dim=1).cpu().numpy()
            target = val_y[:, 0].cpu().numpy()
            val_dice = np.mean([dice(pred_mask[i] != 0, target[i] != 0) for i in range(target.shape[0])])

        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_dice": float(val_dice)}
        history.append(row)
        print("epoch={epoch} loss={loss:.4f} val_dice={val_dice:.4f}".format(**row))

    model.reset()
    model.eval()
    with torch.no_grad():
        prediction = model(val_x[:1], val_y[:1], make_onehot=True, make_onehot_classes=(0, 1))
        samples = model.sample_prior(4, out_device="cpu")

    sample_masks = [torch.argmax(sample[0], dim=0).cpu().numpy() for sample in samples]
    save_qualitative(
        args.out_dir,
        val_x[0, 0].cpu().numpy(),
        val_y[0, 0].cpu().numpy(),
        torch.argmax(prediction[0], dim=0).cpu().numpy(),
        sample_masks,
    )

    torch.save({"model_state_dict": model.state_dict(), "history": history}, os.path.join(args.out_dir, "baseline_checkpoint.pt"))
    np.save(os.path.join(args.out_dir, "history.npy"), np.array(history, dtype=object))


if __name__ == "__main__":
    main()
