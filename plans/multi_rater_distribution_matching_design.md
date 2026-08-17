# Multi-Rater Distribution Matching Design

This note formalizes the current training setup and a proposed alternative that
uses all four LIDC annotations as a mask distribution rather than sampling one
annotation per step. It is meant as project design context, not as a claim about
implemented code.

## 1. Current Problem Setting

Each LIDC training example contains one CT crop and four independent radiologist
segmentations.

Input image:

```text
x in R^{1 x H x W}
```

Human masks:

```text
M = {M_1, M_2, M_3, M_4},   M_g in {0, 1}^{H x W}
```

Current crop size:

```text
H = W = 128
```

The individual masks are binary, but the human disagreement map is continuous.
It is computed from the four masks as normalized binary entropy per pixel:

```text
p_h(y = 1 | x, pixel) = (1 / 4) * sum_g M_g(pixel)
D_h(pixel) = H_binary(p_h(y = 1 | x, pixel))
```

`D_h` is low when all graders agree and high when graders split between lesion
and background. Its values lie in `[0, 1]`; it is not a binary label.

Example values:

```text
0 / 4 graders mark lesion: D_h = 0
1 / 4 graders mark lesion: D_h ~= 0.81
2 / 4 graders mark lesion: D_h = 1
3 / 4 graders mark lesion: D_h ~= 0.81
4 / 4 graders mark lesion: D_h = 0
```

## 2. Current Implemented Model

The implemented model is a Probabilistic U-Net with optional disagreement
supervision.

There are three variants:

```text
baseline: segmentation distribution only
head:     baseline + predicted disagreement head
full:     head + uncertainty/disagreement alignment loss
```

### 2.1 Latent Variable

The latent variable `z` is a low-dimensional random code. In the current runs:

```text
z in R^6
```

Different samples of `z` should produce different plausible segmentation masks
for the same image.

The model has two latent distributions:

```text
prior:     p_theta(z | x)
posterior: q_phi(z | x, M_g)
```

The prior sees only the image and is used at inference time. The posterior sees
the image plus one ground-truth mask and is used only during training.

### 2.2 Current Training Input

The dataset returns all four masks, but the segmentation reconstruction path uses
only one randomly selected grader target per step:

```text
image:  x
masks:  {M_1, M_2, M_3, M_4}
target: M_r, where r is sampled uniformly from {1, 2, 3, 4}
```

The posterior therefore receives:

```text
q_phi(z | x, M_r)
```

It does not receive all four labels at once.

### 2.3 Current Outputs

For all variants:

```text
segmentation log-probabilities: s_theta(x, z) in R^{2 x H x W}
```

For `head` and `full`:

```text
predicted disagreement: D_hat_theta(x) in [0, 1]^{1 x H x W}
```

At inference, the model samples:

```text
z_k ~ p_theta(z | x)
S_k = argmax s_theta(x, z_k)
```

This produces a set of plausible segmentation samples:

```text
S = {S_1, ..., S_K}
```

## 3. Current Losses

### 3.1 Baseline Loss

The baseline trains against one random grader mask:

```text
L_seg = CE(s_theta(x, z), M_r)
L_KL  = KL(q_phi(z | x, M_r) || p_theta(z | x))
```

Total:

```text
L_baseline = L_seg + beta * L_KL
```

In the current experiments:

```text
beta = 1
```

### 3.2 Disagreement Head Loss

For `head` and `full`, the model predicts a disagreement map from U-Net features:

```text
D_hat_theta(x) = sigmoid(head(features_theta(x)))
```

The supervised disagreement loss is:

```text
L_D = MSE(D_hat_theta(x), D_h)
```

The `head` variant uses:

```text
L_head = L_seg + beta * L_KL + lambda_D * L_D
```

### 3.3 Alignment Loss

The `full` model additionally samples multiple segmentations from the prior:

```text
z_k ~ p_theta(z | x)
s_k = s_theta(x, z_k)
```

It computes model uncertainty as entropy of the mean foreground probability:

```text
U_theta(pixel) = H_binary((1 / K) * sum_k P_theta(y = 1 | x, z_k, pixel))
```

Then it aligns that model uncertainty to human disagreement:

```text
L_align = L1(U_theta, D_h)
```

The full loss is:

```text
L_full =
    L_seg
  + beta * L_KL
  + lambda_D * L_D
  + lambda_A * L_align
```

In the most recent fcomb-fixed run:

```text
lambda_D = 0.01
lambda_A = 1e-5
```

## 4. Limitation of the Current Training Loop

The current reconstruction loss sees only one grader target per image per step.
This has two consequences:

1. Three available labels are ignored by the segmentation path on each step.
2. Multi-rater structure reaches the segmentation distribution only indirectly,
   through the disagreement losses and through stochastic random-grader sampling
   across training.

This is simple and close to the original Probabilistic U-Net setup, but it is not
the most direct use of the LIDC label structure.

## 5. Proposed Theoretical Reframing

Instead of viewing the four masks as four interchangeable labels, view them as
samples from a conditional human annotation distribution:

```text
P_human(mask | x)
```

The model also defines a conditional mask distribution:

```text
P_theta(mask | x)
```

The main objective should then be:

```text
match P_theta(mask | x) to P_human(mask | x)
```

This directly matches the project goal: generate a distribution of segmentations
that resembles the distribution of human annotations.

## 6. Empirical Human Mask Distribution

The simplest human distribution is the empirical distribution over four masks:

```text
P_human = Uniform({M_1, M_2, M_3, M_4})
```

However, these four masks should not necessarily be treated as exact atoms. Each
annotation is itself a noisy sample from a broader possible annotation process.

A smoother formulation treats each mask as the center of a local kernel:

```text
P_human(S | x) proportional to (1 / 4) * sum_g K(S, M_g)
```

A useful mask-space kernel is distance-based:

```text
K(S, M_g) = exp(-d(S, M_g) / tau)
```

where `d` can be soft Dice distance or soft IoU distance, and `tau` controls how
strictly a predicted sample must match a human mask.

This is better than a pixelwise Gaussian in raw mask space, because the human
annotations are binary structured masks, not ordinary independent continuous
vectors. This does not mean the disagreement target is binary; `D_h` remains a
continuous entropy map.

## 7. Proposed Model Output

For each image, the model should produce `K` soft segmentation samples:

```text
P_theta = Uniform({S_1, ..., S_K})
S_k in [0, 1]^{H x W}
```

These may come from the existing prior/decoder mechanism:

```text
z_k ~ p_theta(z | x)
S_k = foreground_probability(s_theta(x, z_k))
```

or from a future architecture that directly samples masks. The important object
is the induced distribution over masks, not the Gaussian latent itself.

## 8. Distribution-Matching Loss

A practical differentiable loss should compare the set of model samples to the
set of human masks.

### 8.1 Soft Mask Distance

Use a differentiable soft Dice or soft IoU distance between one model-sampled
soft mask and one human annotation:

```text
d(S_i, M_g)
```

This is only a pairwise building block. Training should not randomly pair each
model sample with one arbitrary grader mask.

```text
d_dice(S, M) = 1 - (2 * sum(S * M) + eps) / (sum(S) + sum(M) + eps)
```

or:

```text
d_iou(S, M) = 1 - (sum(S * M) + eps) / (sum(S + M - S * M) + eps)
```

### 8.2 Bidirectional Coverage Loss

A safe set-matching objective is bidirectional soft coverage:

```text
L_human_covered =
    mean_g softmin_i d(S_i, M_g)

L_model_valid =
    mean_i softmin_g d(S_i, M_g)

L_dist = L_human_covered + L_model_valid
```

`L_human_covered` says every human annotation should be close to at least one
model sample.

`L_model_valid` says every model sample should look like at least one human
annotation, preventing arbitrary diverse noise.

The soft minimum can be implemented with log-sum-exp:

```text
softmin_i a_i = -tau * logsumexp_i(-a_i / tau)
```

### 8.3 Kernel Likelihood View

The same idea can be written as a two-way kernel-density objective. This is the
preferred distribution-matching interpretation: each model sample is scored by
its likelihood under the empirical human mask distribution, and each human mask
is scored by its likelihood under the model sample distribution.

```text
L_model_to_human =
    - mean_i log [(1 / 4) * sum_g exp(-d(S_i, M_g) / tau)]

L_human_to_model =
    - mean_g log [(1 / K) * sum_i exp(-d(S_i, M_g) / tau)]

L_dist = L_model_to_human + L_human_to_model
```

This treats the four labels as centers of a smooth distribution in mask space.
It avoids arbitrary sample-label pairing: a sample receives high likelihood if it
is close to any human mask. The reverse term prevents collapse onto only the
easiest or majority-like annotation by requiring every human mask to be covered
by at least one model sample.

## 9. Relationship to GED

The evaluation metric GED already compares the predicted mask distribution to
the human mask distribution:

```text
GED(P_theta, P_human)
```

For samples:

```text
GED =
    2 * E[d(S, M)]
  - E[d(S, S')]
  - E[d(M, M')]
```

where `S, S'` are model samples and `M, M'` are human masks.

Training directly with a differentiable GED-style loss is possible, but the
model-model diversity term can be risky: if not balanced carefully, the model may
increase diversity without improving anatomical validity. The bidirectional
coverage loss above is usually safer as a first implementation.

## 10. Where Disagreement Fits in the New Framework

In the distribution-matching view, disagreement is not a separate concept. It is
a summary statistic of the human mask distribution.

Human disagreement:

```text
D_h(pixel) = entropy_{M ~ P_human}[M(pixel)]
```

Model uncertainty:

```text
U_theta(pixel) = entropy_{S ~ P_theta}[S(pixel)]
```

If the full mask distributions match, then these entropy maps should match too:

```text
P_theta(mask | x) approx P_human(mask | x)
implies
U_theta approx D_h
```

Therefore, disagreement can serve three roles:

1. Diagnostic metric: does model uncertainty match human disagreement?
2. Auxiliary output: predict `D_h` directly for clinical interpretability.
3. Consistency regularizer: make the explicit disagreement prediction agree with
   uncertainty implied by the model samples.

The disagreement losses in a distribution-matching model could be:

```text
L_D = MSE(D_hat_theta(x), D_h)
L_consistency = MSE(D_hat_theta(x), U_theta)
```

The full proposed objective becomes:

```text
L =
    L_dist(P_theta, P_human)
  + lambda_D * L_D
  + lambda_C * L_consistency
  + optional latent regularization
```

Here, the distribution-matching loss is responsible for making samples match
human annotations. The disagreement head becomes a useful readout and stabilizer,
not the only mechanism forcing sample uncertainty into annotator-disagreement
regions.

## 11. Why Not Remove Reconstruction Entirely?

One tempting idea is to learn only:

```text
KL(q(z | x, all masks) || p(z | x))
```

This is not enough. It only makes two latent distributions agree. It does not
force decoded samples to resemble lesions.

A degenerate solution could satisfy the KL while producing meaningless masks:

```text
q(z | x, labels) = p(z | x)
decoder ignores z or predicts nonsense
```

Some loss must connect predicted masks to human masks:

```text
z -> decoded mask -> comparison with annotations
```

The proposed `L_dist` is that reconstruction signal, generalized from one label
to the whole multi-rater annotation set.

## 12. Suggested Practical Next Step

For a future implementation, the lowest-risk version is:

1. Keep the existing prior and decoder.
2. Sample `K` masks from the prior during training.
3. Compute soft mask probabilities, not hard argmax masks.
4. Compute pairwise soft Dice/IoU distances between `K` model samples and four
   human masks.
5. Optimize the bidirectional coverage loss.
6. Keep the disagreement head loss as an auxiliary output.

First experimental objective:

```text
L =
    L_dist
  + lambda_D * MSE(D_hat, D_h)
```

Then compare against the current full model on:

```text
Dice / IoU
GED
U-D correlation
predicted D correlation
sample diversity
nestedness
```

## 13. Reporting Interpretation

The current model can be described as:

```text
single-rater Probabilistic U-Net training
+ explicit disagreement prediction
+ uncertainty/disagreement alignment
```

The proposed model can be described as:

```text
direct matching between the model's sampled mask distribution
and the empirical multi-rater annotation distribution,
with disagreement as an entropy-derived auxiliary readout.
```

This reframing is stronger theoretically because it makes the four annotations
central to the segmentation objective rather than using them mainly through an
auxiliary disagreement map.
