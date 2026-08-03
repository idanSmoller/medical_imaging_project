# Probabilistic U-Net

This repository contains a generic PyTorch implementation of the
[Probabilistic U-Net](https://arxiv.org/abs/1806.05034) that somewhat mirrors
the signature of the
[official implementation](https://github.com/SimonKohl/probabilistic_unet) in
Tensorflow.

The legacy experiment-management code has been removed so the project can run
with a small, explicit PyTorch training loop suitable for Colab and course
submission.


## Installation

```bash
pip install -e .
```

will make a package called `probunet` available in your current Python environment.


## Probabilistic U-Net

Our generic implementation of the Probabilistic U-Net is hopefully relatively straightforward to use:

```python
from probunet.model import ProbabilisticSegmentationNet
```

As you will have noticed, it also has a pretty generic name. That's because it doesn't actually require a U-Net, but can work with arbitrary segmentation architectures, as long as they:

1. Allow injection of samples in some way.
2. Provide the same signature as our InjectionUNet (look at the calls to self.task_net to see requirements.)

Our encoder implementation also accepts injections, this is currently not used. Make sure to read the method docstrings of the ProbabilisticSegmentationNet, there are some quirks, e.g. `.reconstruct()` doesn't compute gradients.

## Baseline Smoke Training

The repository includes a small plain-PyTorch baseline runner:

```bash
python scripts/baseline_smoke_train.py --epochs 3 --device cpu
```

It trains the original Probabilistic U-Net mechanics on the available local
lesion sequence NPZ by rasterizing bounding boxes into segmentation masks,
then saves:

- `outputs/baseline_smoke/baseline_checkpoint.pt`
- `outputs/baseline_smoke/history.npy`
- `outputs/baseline_smoke/qualitative_samples.png`

For the final disagreement-aware project experiments, prefer adding similarly
explicit PyTorch scripts under `scripts/`.
