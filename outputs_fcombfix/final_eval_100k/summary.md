# LIDC test results (16 samples per image)

Checkpoints from `outputs_fcombfix/lidc_ablation`.

| variant | step | dice | IoU | GED | U-D corr (entropy) | U-D corr (mutual info) | pred D corr | E[d(S,S')] | nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100000 | 0.3552 | 0.2800 | 0.3939 | 0.4828 | 0.4876 | n/a | 0.6865 | 1% |
| head | 100000 | 0.3079 | 0.2407 | 0.4025 | 0.4923 | 0.4889 | 0.6087 | 0.6872 | 3% |
| full | 100000 | 0.3395 | 0.2667 | 0.3884 | 0.5086 | 0.5015 | 0.6270 | 0.6921 | 1% |

**Trivial control for the disagreement head (plan section 22 Q1):** the boundary band of the model's own predicted segmentation correlates with D_GT at **0.4308**. The head's `pred D corr` must beat this to show it learned something about raters rather than about lesion outlines.

`nested` is the fraction of images whose samples are all nested inside one another. A high value means the latent only rescales a single mask, which bounds what any diversity-shaping loss can achieve.
