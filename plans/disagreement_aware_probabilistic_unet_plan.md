# Disagreement-Aware Probabilistic U-Net

## Implementation Plan Based on the Original Probabilistic U-Net Codebase

## 1. Project Goal

The project extends the **Probabilistic U-Net** for multi-rater medical
image segmentation.

The original model learns a distribution over plausible segmentations:

$$
p(y \mid x)
$$

Our extension additionally models **where human annotators disagree**.
Given multiple ground-truth masks for the same image, we construct a
pixel-wise disagreement map $D^{GT}(x)$ and train the model to:

1.  predict this human disagreement map explicitly; and
2.  generate segmentation samples whose spatial variability matches the
    observed human disagreement.

**Central hypothesis:** Explicit supervision from inter-rater
disagreement will make the Probabilistic U-Net's predictive diversity
better aligned with genuine human annotation ambiguity.

The goal is not merely to improve Dice score. The primary goal is to
improve **where and how the model expresses uncertainty**.

------------------------------------------------------------------------

## 2. Base Model

We begin with the official Probabilistic U-Net implementation supplied
for the paper *A Probabilistic U-Net for Segmentation of Ambiguous
Images* by Simon A. A. Kohl et al.

The model consists conceptually of:

-   a U-Net segmentation backbone;
-   a prior network $p(z \mid x)$;
-   a posterior network $q(z \mid x,y)$;
-   a latent variable $z$;
-   an `fcomb` component that combines U-Net features with $z$ to
    generate a segmentation.

During inference, multiple latent samples can produce multiple plausible
segmentations:

$$
z_k \sim p(z \mid x), \qquad S_1,S_2,\ldots,S_K.
$$

------------------------------------------------------------------------

# Phase 1 --- Reproduce the Baseline

## 3. Get the Original Model Running

Before modifying the model:

1.  load the LIDC data;
2.  train the original Probabilistic U-Net;
3.  generate multiple segmentations for the same image;
4.  reproduce the original evaluation pipeline;
5.  save a baseline checkpoint.

Clone the repository:

``` bash
git clone https://github.com/SimonKohl/probabilistic_unet.git
cd probabilistic_unet
```

For a test image, save qualitative examples containing the input image,
all available ground-truth annotations, and several Probabilistic U-Net
samples.

No architectural changes should be made until this pipeline works.

------------------------------------------------------------------------

# Phase 2 --- Dataset

## 4. Use the Preprocessed LIDC Data

Use the cropped LIDC-IDRI data rather than preprocessing the complete
raw DICOM collection.

Each training example should expose the image and all available masks:

``` python
{
    "image": image,
    "masks": [mask1, mask2, mask3, mask4]
}
```

The original Probabilistic U-Net training can still randomly select one
annotation as the target for the normal posterior/reconstruction step.
Our extension additionally uses **all available masks simultaneously**
to calculate human disagreement.

------------------------------------------------------------------------

# Phase 3 --- Human Disagreement Target

## 5. Construct a Pixel-Wise Disagreement Map

For $G$ annotations $Y^{(1)},\ldots,Y^{(G)}$, compute the empirical
foreground probability at pixel $i$:

$$
P_i = \frac{1}{G}\sum_{g=1}^{G}Y_i^{(g)}.
$$

Then calculate normalized binary entropy:

$$
D_i^{GT}
=
-\frac{
P_i\log(P_i+\epsilon)
+
(1-P_i)\log(1-P_i+\epsilon)
}{
\log 2
}.
$$

This gives $D_i^{GT}\in[0,1]$.

-   $D_i^{GT}=0$: all annotators agree.
-   $D_i^{GT}=1$: maximal disagreement.
-   Intermediate values represent partial disagreement.

Example utility:

``` python
def compute_disagreement(masks, eps=1e-6):
    # masks shape: [B, G, H, W]
    p = masks.mean(axis=1, keepdims=True)
    entropy = -(
        p * np.log(p + eps)
        + (1.0 - p) * np.log(1.0 - p + eps)
    )
    return entropy / np.log(2.0)
```

Prefer computing this online from the masks. Before proceeding,
visualize many examples to verify that disagreement maps correspond to
meaningful annotation differences.

------------------------------------------------------------------------

# Phase 4 --- Disagreement Prediction Head

## 6. Add an Auxiliary Disagreement Head

Attach a small head to the final spatial U-Net features $F(x)$:

``` text
                         segmentation pathway
                              ↑
image → U-Net → features ─────┤
                              │
                              └→ 1×1 convolution → sigmoid
                                                    ↓
                                             disagreement map
```

The head predicts:

$$
\hat D(x)=\sigma(W_D * F(x)+b_D).
$$

Start with a single **1×1 convolution** and sigmoid. Do not introduce a
second encoder initially.

## 7. Disagreement Prediction Loss

Train the new head against the ground-truth disagreement map using MSE:

$$
L_D =
\frac{1}{HW}\sum_i(\hat D_i-D_i^{GT})^2.
$$

The first extended objective is:

$$
L=L_{\text{PU-Net}}+\lambda_D L_D.
$$

Test a small set such as $\lambda_D\in\{0.1,0.5,1.0\}$.

This is an intermediate ablation: it predicts disagreement but does not
yet force stochastic segmentation samples to express diversity in those
locations.

------------------------------------------------------------------------

# Phase 5 --- Align Model Diversity With Human Disagreement

## 8. Generate Multiple Samples During Training

For each image, draw $K$ samples:

$$
z_1,\ldots,z_K\sim p(z\mid x)
$$

and generate foreground probability maps:

$$
Q^{(1)},\ldots,Q^{(K)}.
$$

Start with $K=4$ during training. At test time, use 16 or 32 samples.

## 9. Construct a Model Uncertainty Map

Calculate the mean foreground probability:

$$
\bar Q_i=\frac{1}{K}\sum_{k=1}^{K}Q_i^{(k)}.
$$

Then calculate normalized binary entropy:

$$
U_i=
-\frac{
\bar Q_i\log(\bar Q_i+\epsilon)
+
(1-\bar Q_i)\log(1-\bar Q_i+\epsilon)
}{
\log 2
}.
$$

Here:

-   $D^{GT}$ = observed **human disagreement**;
-   $U$ = **uncertainty/diversity expressed by model samples**.

Keep these concepts distinct throughout the implementation and report.

## 10. Disagreement Alignment Loss

Encourage model uncertainty to spatially match human disagreement:

$$
L_{\text{align}}
=
\frac{1}{HW}\sum_i|U_i-D_i^{GT}|.
$$

The full objective becomes:

$$
L_{\text{total}}
=
L_{\text{PU-Net}}
+
\lambda_D L_D
+
\lambda_A L_{\text{align}}.
$$

The intended behavior is:

> Generated diversity should occur primarily in regions where human
> annotators disagree.

------------------------------------------------------------------------

# Phase 6 --- Preserve the Original Latent Model

## 11. Do Not Initially Condition the Prior on Disagreement

Keep the original latent structure:

$$
p(z\mid x), \qquad q(z\mid x,y).
$$

Do not initially change the prior to $p(z\mid x,\hat D)$. Since
$\hat D=f(x)$, it does not provide fundamentally new input information
and would make the ablation harder to interpret.

Instead, the disagreement head affects the shared representation through
auxiliary supervision, while the alignment loss directly constrains the
distribution of generated segmentations.

------------------------------------------------------------------------

# Phase 7 --- Training Procedure

## 12. One Training Iteration

1.  Load $x,Y_1,\ldots,Y_G$.
2.  Compute $D^{GT}$ from all annotations.
3.  Randomly choose one $Y_g$ as the standard segmentation target.
4.  Run the original Probabilistic U-Net forward pass and compute
    $L_{\text{PU-Net}}$.
5.  Predict $\hat D=f_D(F(x))$ and compute $L_D$.
6.  Draw $K$ samples $z_k\sim p(z\mid x)$ and generate
    $Q^{(1)},\ldots,Q^{(K)}$.
7.  Compute the model uncertainty map $U$.
8.  Compute $L_{\text{align}}=\|U-D^{GT}\|_1$.
9.  Optimize:

$$
L_{\text{total}}
=
L_{\text{PU-Net}}
+
\lambda_D L_D
+
\lambda_A L_{\text{align}}.
$$

------------------------------------------------------------------------

# Phase 8 --- Experiments and Ablations

## 13. Required Models

Train at least three variants:

  Model                             Disagreement head   Alignment loss
  ------------------------------- ------------------- ----------------
  Original Probabilistic U-Net                     No               No
  \+ disagreement supervision                     Yes               No
  Full disagreement-aware model                   Yes              Yes

### Model A --- Original Probabilistic U-Net

$$
L=L_{\text{PU-Net}}.
$$

### Model B --- Disagreement Prediction Only

$$
L=L_{\text{PU-Net}}+\lambda_D L_D.
$$

This tests whether human-disagreement prediction works as a useful
auxiliary task.

### Model C --- Full Model

$$
L=L_{\text{PU-Net}}+\lambda_D L_D+\lambda_A L_{\text{align}}.
$$

This tests the complete hypothesis.

If resources permit, test a small number of $\lambda_A$ values, such as
0.1, 0.5, and 1.0. Avoid a large hyperparameter search.

------------------------------------------------------------------------

# Phase 9 --- Evaluation

## 14. Segmentation Quality

Measure standard segmentation quality:

-   Dice;
-   IoU.

These metrics verify that disagreement-aware training does not
substantially damage segmentation quality. The main claim should not
depend only on improved Dice.

## 15. Distributional Quality

Preserve the original Probabilistic U-Net evaluation where possible,
particularly the **generalized energy distance** used to compare
generated segmentation distributions with human annotation
distributions.

## 16. New Disagreement-Alignment Metrics

For every test image:

1.  calculate human disagreement $D^{GT}$;
2.  generate 16--32 model samples;
3.  calculate model uncertainty $U^{model}$;
4.  compare them.

### Mean Absolute Error

$$
\text{MAE}
=
\frac{1}{HW}\sum_i|U_i^{model}-D_i^{GT}|.
$$

Lower is better.

### Correlation

Calculate the correlation between $U^{model}$ and $D^{GT}$.

Higher correlation means the model is uncertain in the same spatial
locations where human annotators disagree.

This should be one of the project's primary quantitative results.

------------------------------------------------------------------------

# Phase 10 --- Qualitative Evaluation

## 17. Visualization

For selected test cases, create figures containing:

``` text
Input CT image

Human masks
M1   M2   M3   M4

Human disagreement map
D_GT

Original Probabilistic U-Net samples
S1   S2   S3   S4

Baseline model uncertainty
U_baseline

Full-model samples
S1   S2   S3   S4

Full-model uncertainty
U_ours

Predicted human disagreement
D_hat
```

Look especially for examples where humans agree in the lesion interior
but disagree near a boundary, and compare whether the baseline and
proposed model place stochastic variation in those disputed areas.

------------------------------------------------------------------------

# Phase 11 --- Proposed Code Organization

## 18. Keep the Baseline Intact

Avoid heavily modifying the baseline implementation. Prefer new
files/classes that reuse existing components:

``` text
probabilistic_unet/
│
├── model/
│   ├── ...
│   └── disagreement_head.py
│
├── data/
│   ├── ...
│   └── lidc_disagreement.py
│
├── training/
│   ├── train_prob_unet.py
│   └── train_disagreement_prob_unet.py
│
├── evaluation/
│   ├── ...
│   └── eval_disagreement.py
│
└── utils/
    └── disagreement.py
```

The original training script should remain usable for reproducing the
baseline.

------------------------------------------------------------------------

# Phase 12 --- Implementation Milestones

## Milestone 1 --- Baseline

**Estimated effort: 1--2 days**

-   Download preprocessed LIDC data.
-   Configure the data loader.
-   Run original Probabilistic U-Net training.
-   Generate segmentation samples.
-   Verify that loss decreases.
-   Run baseline evaluation.
-   Save checkpoints and example outputs.

## Milestone 2 --- Ground-Truth Disagreement

**Estimated effort: half a day**

Implement:

$$
Y_1,\ldots,Y_G\rightarrow D^{GT}.
$$

Visualize 30--50 examples and verify that the maps correspond to
meaningful annotation differences.

## Milestone 3 --- Disagreement Prediction Head

**Estimated effort: 1 day**

Add:

$$
F(x)\rightarrow\hat D.
$$

Train with:

$$
L=L_{\text{PU-Net}}+\lambda_D L_D.
$$

Evaluate whether $\hat D\approx D^{GT}$.

## Milestone 4 --- Sample-Alignment Loss

**Estimated effort: 1--2 days**

Generate $K=4$ prior samples during training, calculate $U^{model}$, add
$L_{\text{align}}$, and verify gradient flow and training stability.

## Milestone 5 --- Ablation Experiments

**Estimated effort: 1--2 days**

Train:

1.  baseline Probabilistic U-Net;
2.  Probabilistic U-Net + disagreement head;
3.  full disagreement-aware model.

Optionally test a small number of loss-weight settings.

## Milestone 6 --- Final Evaluation

**Estimated effort: 1 day**

-   Generate 16--32 samples per test image.
-   Calculate Dice/IoU.
-   Calculate generalized energy distance.
-   Calculate disagreement MAE.
-   Calculate disagreement correlation.
-   Create qualitative comparison figures.

------------------------------------------------------------------------

# 19. Key Terminology

Maintain a clear distinction between:

### Human disagreement

$$
D^{GT}(x)
$$

Computed directly from multiple human annotations.

### Predicted human disagreement

$$
\hat D(x)
$$

Predicted by the auxiliary disagreement head from the input image.

### Model uncertainty

$$
U(x)
$$

Computed from the variability of multiple segmentation samples generated
by the Probabilistic U-Net.

The project asks whether:

$$
\hat D(x)\approx D^{GT}(x)
$$

and, more importantly,

$$
U(x)\approx D^{GT}(x).
$$

In words:

> Can the model both recognize where humans disagree and express its
> segmentation uncertainty in those same locations?

------------------------------------------------------------------------

# 20. Expected Contribution

The original Probabilistic U-Net learns a distribution of plausible
segmentations from ambiguous annotations.

Our extension adds **explicit spatial supervision of inter-rater
disagreement**.

The proposed model is trained not only to produce diverse plausible
segmentations, but to place that diversity in regions where the human
annotations indicate genuine ambiguity.

The intended contribution is:

> A disagreement-aware training strategy for Probabilistic U-Net that
> aligns spatial predictive diversity with observed inter-rater
> disagreement.

This should be presented as an extension of the original Probabilistic
U-Net, not as a claim that expert-disagreement modeling itself is
unprecedented.

------------------------------------------------------------------------

# 21. Practical Note About the Original Repository

The official repository is an older TensorFlow-based implementation and
may require compatibility work in a current Google Colab environment.

Therefore, establish a working baseline before implementing
disagreement-aware components.

If maintaining the historical implementation becomes a major obstacle,
consider a faithful modern reimplementation if permitted by the course
requirements.

For a controlled scientific comparison, keep:

-   the same dataset;
-   the same train/validation/test split;
-   the same base Probabilistic U-Net architecture;
-   the same training configuration where possible;
-   extension-specific changes isolated and documented.

------------------------------------------------------------------------

# 22. Final Experimental Story

The final report should answer three questions:

1.  **Can the model predict where human annotators disagree?**

    Compare $\hat D$ with $D^{GT}$.

2.  **Does disagreement-aware training improve the spatial calibration
    of stochastic segmentation diversity?**

    Compare $U^{model}$ with $D^{GT}$.

3.  **Does this improvement preserve the quality of the segmentation
    distribution?**

    Compare baseline and proposed models using Dice/IoU and generalized
    energy distance.

The desired result is not simply:

> Our Dice score is higher.

The stronger result is:

> The original Probabilistic U-Net generates diverse plausible
> segmentations, while our disagreement-aware extension makes that
> diversity better correspond to the regions where human experts
> actually disagree.
