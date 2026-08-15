import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import distributions
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from probunet.disagreement import (
    compute_disagreement,
    model_uncertainty_from_samples,
)
from probunet.eval import (
    dice,
    disagreement_correlation,
    disagreement_mae,
    generalized_energy_distance,
    jaccard,
)
from probunet.lidc import LIDCCrops
from probunet.model import (
    DisagreementAwareProbabilisticSegmentationNet,
    InjectionConvEncoder,
    InjectionUNet,
    ProbabilisticSegmentationNet,
)


VARIANTS = ("baseline", "head", "full")


def limit_dataset(dataset, max_items):
    if max_items is None or max_items >= len(dataset):
        return dataset
    return Subset(dataset, range(max_items))


def make_loaders(args):
    train_set = LIDCCrops(
        root=args.data_root,
        split="train",
        crop_size=args.crop_size,
        single_random_grader=True,
    )
    val_set = LIDCCrops(
        root=args.data_root,
        split=args.eval_split,
        crop_size=args.crop_size,
        train=False,
        single_random_grader=True,
    )
    train_set = limit_dataset(train_set, args.max_train)
    val_set = limit_dataset(val_set, args.max_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return train_loader, val_loader


def make_model(args):
    model_cls = (
        ProbabilisticSegmentationNet
        if args.variant == "baseline"
        else DisagreementAwareProbabilisticSegmentationNet
    )
    return model_cls(
        in_channels=1,
        out_channels=2,
        num_feature_maps=args.feature_maps,
        latent_size=args.latent_size,
        depth=args.depth,
        latent_distribution=distributions.Normal,
        task_op=InjectionUNet,
        task_kwargs={
            "injection_channels": args.latent_size,
            "output_activation_op": nn.LogSoftmax,
            "output_activation_kwargs": {"dim": 1},
            "activation_kwargs": {"inplace": True},
        },
        prior_op=InjectionConvEncoder,
        prior_kwargs={
            "in_channels": 1,
            "out_channels": 2 * args.latent_size,
            "depth": args.depth,
            "num_feature_maps": args.feature_maps,
            "activation_kwargs": {"inplace": True},
            "norm_depth": 0,
        },
        posterior_op=InjectionConvEncoder,
        posterior_kwargs={
            "in_channels": 3,
            "out_channels": 2 * args.latent_size,
            "depth": args.depth,
            "num_feature_maps": args.feature_maps,
            "activation_kwargs": {"inplace": True},
            "norm_depth": 0,
        },
    ).to(args.device)


def kl_loss(model):
    kl = distributions.kl_divergence(model.posterior, model.prior)
    return kl.reshape(kl.shape[0], -1).sum(dim=1).mean()


def set_lr(optimizer, step, args):
    if args.lr_final >= args.lr:
        return args.lr
    decay_every = max(1, args.steps // args.lr_decay_steps)
    decay_index = min((step - 1) // decay_every, args.lr_decay_steps)
    gamma = (args.lr_final / args.lr) ** (1.0 / args.lr_decay_steps)
    lr = args.lr * (gamma ** decay_index)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_step(model, batch, optimizer, criterion, args, step):
    model.train()
    model.reset()
    optimizer.zero_grad()
    lr = set_lr(optimizer, step, args)

    image = batch["image"]
    target = batch["target"].long()
    masks = batch["masks"]

    prediction = model(image, target, make_onehot=True, make_onehot_classes=(0, 1))
    loss_seg = criterion(prediction, target[:, 0].long())
    loss_kl = kl_loss(model)
    loss = loss_seg + args.beta * loss_kl

    log = {
        "step": step,
        "lr": lr,
        "loss": loss.detach(),
        "loss_seg": loss_seg.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_disagreement": prediction.new_tensor(0.0),
        "loss_alignment": prediction.new_tensor(0.0),
    }

    if args.variant in ("head", "full"):
        lambda_alignment = args.lambda_alignment if args.variant == "full" else 0.0
        loss_aux, aux_metrics = model.disagreement_losses(
            image,
            masks,
            n_samples=args.train_samples,
            lambda_disagreement=args.lambda_disagreement,
            lambda_alignment=lambda_alignment,
        )
        loss = loss + loss_aux
        log["loss"] = loss.detach()
        log["loss_disagreement"] = aux_metrics["loss_disagreement"]
        log["loss_alignment"] = aux_metrics["loss_alignment"]

    loss.backward()
    # The reconstruction NLL is summed over batch *and* pixels, so gradients are
    # ~32x the paper's per-image convention. Without this, all three variants NaN
    # out in the prior encoder within ~2k steps once the fcomb nonlinearity is in.
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in log.items()
    }


def sample_masks(model, image, n_samples):
    outputs = model.sample_prior(n_samples, out_device=image.device, input_=image)
    outputs = torch.stack(outputs, dim=0)
    masks = torch.argmax(outputs, dim=2)
    return outputs, masks


def evaluate(model, loader, args):
    model.eval()
    dice_scores = []
    iou_scores = []
    ged_scores = []
    disagreement_maes = []
    disagreement_corrs = []
    predicted_maes = []
    predicted_corrs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
            batch = batch_to_device(batch, args.device)
            image = batch["image"]
            masks = batch["masks"]

            model.reset()
            prediction = model(image)
            pred_mask = torch.argmax(prediction, dim=1).cpu().numpy()
            grader_masks = masks.cpu().numpy().astype(bool)

            outputs, prior_masks = sample_masks(model, image, args.eval_samples)
            model_uncertainty = model_uncertainty_from_samples(outputs).cpu().numpy()
            human_disagreement = compute_disagreement(masks).cpu().numpy()

            predicted_disagreement = None
            if args.variant in ("head", "full"):
                predicted_disagreement = model.predict_disagreement().cpu().numpy()

            prior_masks = prior_masks.cpu().numpy().astype(bool)
            for i in range(image.shape[0]):
                for grader in range(grader_masks.shape[1]):
                    dice_scores.append(
                        dice(pred_mask[i] != 0, grader_masks[i, grader], nan_for_nonexisting=True)
                    )
                    iou_scores.append(
                        jaccard(pred_mask[i] != 0, grader_masks[i, grader], nan_for_nonexisting=True)
                    )
                ged_scores.append(generalized_energy_distance(prior_masks[:, i], grader_masks[i]))
                disagreement_maes.append(
                    disagreement_mae(model_uncertainty[i, 0], human_disagreement[i, 0])
                )
                disagreement_corrs.append(
                    disagreement_correlation(model_uncertainty[i, 0], human_disagreement[i, 0])
                )
                if predicted_disagreement is not None:
                    predicted_maes.append(
                        disagreement_mae(predicted_disagreement[i, 0], human_disagreement[i, 0])
                    )
                    predicted_corrs.append(
                        disagreement_correlation(predicted_disagreement[i, 0], human_disagreement[i, 0])
                    )

    result = {
        "val_dice": float(np.nanmean(dice_scores)),
        "val_iou": float(np.nanmean(iou_scores)),
        "val_ged": float(np.nanmean(ged_scores)),
        "val_uncertainty_mae": float(np.nanmean(disagreement_maes)),
        "val_uncertainty_corr": float(np.nanmean(disagreement_corrs)),
    }
    if predicted_maes:
        result["val_predicted_disagreement_mae"] = float(np.nanmean(predicted_maes))
        result["val_predicted_disagreement_corr"] = float(np.nanmean(predicted_corrs))
    return result


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, step, args, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device, load_optimizer=True):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint


def best_ged_from_history(path):
    if not os.path.exists(path):
        return None
    values = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("val_ged") not in (None, ""):
                values.append(float(row["val_ged"]))
    if not values:
        return None
    return min(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--data-root", default="data/lidc")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=240000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-maps", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=6)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--lr-decay-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-disagreement", type=float, default=0.5)
    parser.add_argument("--lambda-alignment", type=float, default=0.5)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=100.0,
                        help="Max gradient norm; 0 disables clipping.")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights from --resume but start a fresh optimizer.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join("outputs", "lidc_ablation", args.variant)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = make_loaders(args)
    model = make_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.NLLLoss(reduction="sum")
    start_step = 1
    history_path = os.path.join(args.out_dir, "history.csv")
    best_ged = best_ged_from_history(history_path)

    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            args.device,
            load_optimizer=not args.reset_optimizer,
        )
        resumed_step = int(checkpoint.get("step", 0))
        start_step = resumed_step + 1
        if best_ged is None:
            metrics = checkpoint.get("metrics") or {}
            best_ged = metrics.get("val_ged")
        print(
            "resumed {} from step {}; next step is {}".format(
                args.resume,
                resumed_step,
                start_step,
            )
        )

    print(
        "{}: {} train / {} {} samples; writing to {}".format(
            args.variant,
            len(train_loader.dataset),
            len(val_loader.dataset),
            args.eval_split,
            args.out_dir,
        )
    )

    if start_step > args.steps:
        print(
            "checkpoint is already at step {}; target --steps is {}, so nothing to do".format(
                start_step - 1,
                args.steps,
            )
        )
        return

    train_iter = iter(train_loader)
    start = time.time()

    progress = tqdm(
        range(start_step, args.steps + 1),
        desc="train",
        dynamic_ncols=True,
        initial=start_step - 1,
        total=args.steps,
    )
    for step in progress:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch_to_device(batch, args.device)
        row = train_step(model, batch, optimizer, criterion, args, step)
        progress.set_postfix(
            loss="{:.4f}".format(row["loss"]),
            seg="{:.4f}".format(row["loss_seg"]),
            kl="{:.4f}".format(row["loss_kl"]),
            lr="{:.2e}".format(row["lr"]),
        )

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_eval:
            metrics = evaluate(model, val_loader, args)
            row.update(metrics)
            row["elapsed_min"] = (time.time() - start) / 60.0
            append_csv(history_path, row)
            tqdm.write(
                "step={step} loss={loss:.4f} dice={val_dice:.4f} "
                "iou={val_iou:.4f} ged={val_ged:.4f} "
                "unc_mae={val_uncertainty_mae:.4f}".format(**row)
            )
            if best_ged is None or metrics["val_ged"] < best_ged:
                best_ged = metrics["val_ged"]
                save_checkpoint(
                    os.path.join(args.out_dir, "best_checkpoint.pt"),
                    model,
                    optimizer,
                    step,
                    args,
                    metrics,
                )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                os.path.join(args.out_dir, "latest_checkpoint.pt"),
                model,
                optimizer,
                step,
                args,
                row,
            )


if __name__ == "__main__":
    main()
