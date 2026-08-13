import math

import numpy as np
import torch
import torch.nn.functional as F


def _binary_entropy(probability, eps=1e-6):
    probability = probability.clamp(eps, 1.0 - eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    return entropy / math.log(2.0)


def compute_disagreement(masks, eps=1e-6):
    """Compute normalized binary entropy from multiple binary annotations.

    Args:
        masks: Tensor or ndarray with shape ``(B, G, *spatial)`` where ``G`` is
            the number of raters/annotations.
        eps: Numerical stability constant.

    Returns:
        Disagreement map with shape ``(B, 1, *spatial)`` and values in [0, 1].
    """

    if torch.is_tensor(masks):
        probability = masks.float().mean(dim=1, keepdim=True)
        return _binary_entropy(probability, eps=eps)

    masks = np.asarray(masks, dtype=np.float32)
    probability = masks.mean(axis=1, keepdims=True)
    probability = np.clip(probability, eps, 1.0 - eps)
    entropy = -(
        probability * np.log(probability)
        + (1.0 - probability) * np.log(1.0 - probability)
    )
    return entropy / np.log(2.0)


def foreground_probability(logits_or_log_probs, foreground_channel=1):
    """Return foreground probabilities from binary or multiclass model output."""

    if logits_or_log_probs.shape[1] == 1:
        return torch.sigmoid(logits_or_log_probs)
    return F.softmax(logits_or_log_probs, dim=1)[:, foreground_channel:foreground_channel + 1]


def model_uncertainty_from_samples(samples, foreground_channel=1, eps=1e-6, sample_dim=0):
    """Compute normalized entropy from multiple segmentation probability maps.

    Args:
        samples: Tensor containing logits/log-probabilities.
        foreground_channel: Channel to treat as foreground when ``C > 1``.
        eps: Numerical stability constant.
        sample_dim: Dimension that indexes stochastic samples.

    Returns:
        Model uncertainty map with shape ``(B, 1, *spatial)``.
    """

    if samples.dim() < 5:
        raise ValueError("Expected samples with shape (K, B, C, *spatial) or (B, K, C, *spatial).")

    if sample_dim != 0:
        samples = samples.movedim(sample_dim, 0)

    if samples.shape[2] == 1:
        probabilities = torch.sigmoid(samples)
    else:
        probabilities = F.softmax(samples, dim=2)[:, :, foreground_channel:foreground_channel + 1]
    mean_probability = probabilities.mean(dim=0)
    return _binary_entropy(mean_probability, eps=eps)


def disagreement_alignment_loss(model_uncertainty, human_disagreement):
    """L1 between model uncertainty and human disagreement.

    reduction="sum" to match the summed reconstruction NLL -- see the note in
    ``DisagreementAwareProbabilisticSegmentationNet.disagreement_losses``.
    """

    return F.l1_loss(model_uncertainty, human_disagreement.float(), reduction="sum")
