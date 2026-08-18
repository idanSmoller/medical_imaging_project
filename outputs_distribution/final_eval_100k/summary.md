# LIDC test results (16 samples per image)

Checkpoints from `outputs_distribution/lidc_ablation`.

| variant | step | dice | IoU | GED | U-D corr (entropy) | U-D corr (mutual info) | pred D corr | E[d(S,S')] | nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage | 100000 | 0.4383 | 0.3688 | 0.7135 | 0.1201 | 0.1152 | 0.5645 | 0.0239 | 100% |
| kernel | 100000 | 0.4051 | 0.3367 | 0.9329 | 0.1033 | 0.0991 | 0.5689 | 0.0004 | 100% |

**Trivial control for the disagreement head (plan section 22 Q1):** the boundary band of the model's own predicted segmentation correlates with D_GT at **0.4662**. The head's `pred D corr` must beat this to show it learned something about raters rather than about lesion outlines.

`nested` is the fraction of images whose samples are all nested inside one another. A high value means the latent only rescales a single mask, which bounds what any diversity-shaping loss can achieve.
