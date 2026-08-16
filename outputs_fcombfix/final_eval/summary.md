# LIDC test results (16 samples per image)

Checkpoints from `outputs_fcombfix/lidc_ablation`.

| variant | step | dice | IoU | GED | U-D corr (entropy) | U-D corr (mutual info) | pred D corr | E[d(S,S')] | nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1000 | 0.3201 | 0.2585 | 0.3886 | 0.4346 | 0.4327 | n/a | 0.6426 | 98% |
| head | 3000 | 0.1802 | 0.1325 | 0.3993 | 0.4677 | 0.4759 | 0.3268 | 0.6288 | 93% |
| full | 71000 | 0.3355 | 0.2638 | 0.3848 | 0.5099 | 0.5010 | 0.6247 | 0.6865 | 2% |

**Trivial control for the disagreement head (plan section 22 Q1):** the boundary band of the model's own predicted segmentation correlates with D_GT at **0.5001**. The head's `pred D corr` must beat this to show it learned something about raters rather than about lesion outlines.

`nested` is the fraction of images whose samples are all nested inside one another. A high value means the latent only rescales a single mask, which bounds what any diversity-shaping loss can achieve.
