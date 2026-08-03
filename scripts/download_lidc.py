"""Download the preprocessed LIDC-IDRI 2D crops used by the Probabilistic U-Net.

The data is DeepMind's release of the LIDC crops (CC BY 3.0), published alongside the
Hierarchical Probabilistic U-Net and linked from the official Probabilistic U-Net repo.
Preprocessing matches Kohl et al. 2018, Appendix H.1: CT scans resampled to 0.5mm x 0.5mm
in-plane, 180 x 180 crops centered on abnormalities, four graders per crop.

Expected result: 8843 / 1993 / 1980 images over 530 / 111 / 103 patients.
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request


BASE_URL = "https://storage.googleapis.com/hpunet-data/lidc_crops"
SPLITS = ("train", "val", "test")

# (images, patients) as documented by the release, used to validate the download.
EXPECTED = {
    "train": (8843, 530),
    "val": (1993, 111),
    "test": (1980, 103),
}


def download(url, destination):
    """Stream a URL to disk, reporting progress on a single line."""

    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        "\r  {:.1f}/{:.1f} MB ({:.0f}%)".format(
                            downloaded / 1e6, total / 1e6, 100 * downloaded / total
                        ),
                        end="",
                    )
        print()

    if total and downloaded != total:
        raise IOError("Truncated download: got {} of {} bytes".format(downloaded, total))


def extract(archive_path, split_dir):
    """Extract the archive so that ``split_dir`` directly contains images/ and gt/.

    The top-level directory name inside the archives is not documented, so it is
    detected rather than assumed.
    """

    with tempfile.TemporaryDirectory(dir=os.path.dirname(split_dir)) as staging:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(staging)

        root = staging
        while True:
            entries = os.listdir(root)
            if "images" in entries and "gt" in entries:
                break
            subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
            if len(subdirs) != 1:
                raise IOError(
                    "Could not locate images/ and gt/ inside {} (found {})".format(
                        archive_path, entries
                    )
                )
            root = os.path.join(root, subdirs[0])

        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        shutil.move(root, split_dir)


def count_split(split_dir):
    """Return (number of images, number of patients) for an extracted split."""

    images_dir = os.path.join(split_dir, "images")
    patients = [p for p in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, p))]
    images = sum(
        len([f for f in os.listdir(os.path.join(images_dir, p)) if f.endswith(".png")])
        for p in patients
    )
    return images, len(patients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/lidc", help="Directory to download into.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete the tar.gz files.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    ok = True

    for split in args.splits:
        split_dir = os.path.join(args.dest, split)
        archive_path = os.path.join(args.dest, "{}.tar.gz".format(split))

        if os.path.isdir(split_dir) and not args.force:
            print("{}: already extracted".format(split))
        else:
            if not os.path.isfile(archive_path) or args.force:
                print("{}: downloading".format(split))
                download("{}/{}.tar.gz".format(BASE_URL, split), archive_path)
            else:
                print("{}: archive already present".format(split))
            print("{}: extracting".format(split))
            extract(archive_path, split_dir)
            if not args.keep_archives:
                os.remove(archive_path)

        images, patients = count_split(split_dir)
        expected_images, expected_patients = EXPECTED[split]
        status = "OK" if (images, patients) == (expected_images, expected_patients) else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(
            "{}: {} images, {} patients (expected {} / {}) [{}]".format(
                split, images, patients, expected_images, expected_patients, status
            )
        )

    if not ok:
        print("\nCounts differ from the published release - the download may be incomplete.")
        return 1
    print("\nData ready under {}".format(os.path.abspath(args.dest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
