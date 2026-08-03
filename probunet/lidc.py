"""Dataset for the preprocessed LIDC-IDRI 2D crops (see scripts/download_lidc.py).

Each sample is a 180 x 180 lung CT crop centered on an abnormality, together with the
binary masks of four independent graders. Up to three of the four masks can be empty,
because graders disagree on whether an abnormality is present at all -- that disagreement
is the signal the disagreement-aware model is meant to learn.

On-disk layout produced by scripts/download_lidc.py::

    <root>/<split>/images/<patient>/z-<z position>_c<crop>.png
    <root>/<split>/gt/<patient>/z-<z position>_c<crop>_l<grader 0-3>.png
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


NUM_GRADERS = 4
SPLITS = ("train", "val", "test")


def _load_png(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"))


class LIDCCrops(Dataset):
    """LIDC 2D crops with all four grader annotations exposed.

    Args:
        root: Directory containing the ``train``/``val``/``test`` subdirectories.
        split: Which split to load.
        crop_size: Output tile size. The paper (Appendix H.1) trains on randomly
            translated 128 x 128 crops of the 180 x 180 tiles. Pass ``None`` to keep
            the full 180 x 180 tile.
        train: Whether to take a random crop (``True``) or a center crop (``False``).
            Defaults to ``split == "train"``.
        single_random_grader: Additionally return ``target``, one uniformly chosen
            grader mask. This is how the baseline Probabilistic U-Net draws
            image-grader pairs during training. ``masks`` is returned either way.
    """

    def __init__(
        self,
        root="data/lidc",
        split="train",
        crop_size=128,
        train=None,
        single_random_grader=False,
    ):
        if split not in SPLITS:
            raise ValueError("split must be one of {}, got {!r}".format(SPLITS, split))

        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.train = (split == "train") if train is None else train
        self.single_random_grader = single_random_grader

        self.images_dir = os.path.join(root, split, "images")
        self.gt_dir = os.path.join(root, split, "gt")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(
                "No LIDC data at {}. Run: python scripts/download_lidc.py".format(
                    os.path.abspath(os.path.join(root, split))
                )
            )

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError("Found no usable samples under {}".format(self.images_dir))

    def _build_index(self):
        """Index (patient, stem) pairs that have an image and all four grader masks."""

        samples = []
        incomplete = 0
        for patient in sorted(os.listdir(self.images_dir)):
            patient_images = os.path.join(self.images_dir, patient)
            if not os.path.isdir(patient_images):
                continue
            patient_gt = os.path.join(self.gt_dir, patient)
            for filename in sorted(os.listdir(patient_images)):
                if not filename.endswith(".png"):
                    continue
                stem = filename[: -len(".png")]
                mask_paths = [
                    os.path.join(patient_gt, "{}_l{}.png".format(stem, grader))
                    for grader in range(NUM_GRADERS)
                ]
                if not all(os.path.isfile(path) for path in mask_paths):
                    incomplete += 1
                    continue
                samples.append((os.path.join(patient_images, filename), mask_paths))

        if incomplete:
            print(
                "LIDCCrops[{}]: skipped {} crops without all {} grader masks".format(
                    self.split, incomplete, NUM_GRADERS
                )
            )
        return samples

    def _crop_origin(self, height, width, size):
        if not self.train:
            return (height - size) // 2, (width - size) // 2
        top = int(torch.randint(0, height - size + 1, (1,)).item())
        left = int(torch.randint(0, width - size + 1, (1,)).item())
        return top, left

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_paths = self.samples[index]

        image = _load_png(image_path).astype(np.float32) / 255.0
        masks = np.stack([(_load_png(path) > 0).astype(np.float32) for path in mask_paths])

        height, width = image.shape
        if self.crop_size is not None:
            size = self.crop_size
            if size > height or size > width:
                raise ValueError(
                    "crop_size {} exceeds tile size {}x{}".format(size, height, width)
                )
            # The same offset is applied to the image and to all four masks.
            top, left = self._crop_origin(height, width, size)
            image = image[top:top + size, left:left + size]
            masks = masks[:, top:top + size, left:left + size]

        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(image))[None],
            "masks": torch.from_numpy(np.ascontiguousarray(masks)),
        }
        if self.single_random_grader:
            grader = int(torch.randint(0, NUM_GRADERS, (1,)).item())
            sample["target"] = sample["masks"][grader][None]
            sample["grader"] = grader
        return sample
