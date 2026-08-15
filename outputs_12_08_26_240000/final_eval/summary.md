# LIDC test results (16 samples per image)

Checkpoints from `outputs_12_08_26_240000/lidc_ablation`.

| variant | step | dice | IoU | GED | U-D corr (entropy) | U-D corr (mutual info) | pred D corr | E[d(S,S')] | nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 41000 | 0.3269 | 0.2617 | 0.3161 | 0.5841 | 0.5773 | n/a | 0.5833 | 100% |
| head | 35000 | 0.3283 | 0.2672 | 0.3137 | 0.5966 | 0.5904 | 0.6172 | 0.5658 | 100% |
| full | 28000 | 0.2885 | 0.2304 | 0.3298 | 0.6079 | 0.5996 | 0.5950 | 0.5629 | 100% |

**Trivial control for the disagreement head (plan section 22 Q1):** the boundary band of the model's own predicted segmentation correlates with D_GT at **0.5453**. The head's `pred D corr` must beat this to show it learned something about raters rather than about lesion outlines.

`nested` is the fraction of images whose samples are all nested inside one another. A high value means the latent only rescales a single mask, which bounds what any diversity-shaping loss can achieve.
