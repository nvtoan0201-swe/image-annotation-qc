#!/usr/bin/env python3
"""Download the raw image pool (coco128) and optionally the COCO reference labels.

The archive checksum is pinned in configs/classes.yaml, so a corrupted or
changed download is detected instead of silently entering the pipeline.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from annotation_qc.io import (
    find_project_root,
    human_size,
    load_config,
    sha256_file,
)

IMAGE_MEMBER_PREFIX = "coco128/images/train2017/"
LABEL_MEMBER_PREFIX = "coco128/labels/train2017/"


def download_archive(url: str, destination: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 256):
            handle.write(chunk)


def extract_members(archive_path: Path, prefix: str, destination: Path, suffix: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and name.endswith(suffix) and not name.startswith("__MACOSX"):
                member_name = Path(name).name
                target = destination / member_name
                with archive.open(name) as source, target.open("wb") as out:
                    out.write(source.read())
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if images exist")
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="also extract the original coco128 labels into data/reference/labels",
    )
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    dataset = config["dataset"]
    paths = config["paths"]
    raw_images = root / paths["raw_images"]

    existing = len(list(raw_images.glob("*.jpg"))) if raw_images.is_dir() else 0
    print("Dataset download")
    print(f"  source      : {dataset['source']}")
    print(f"  license     : {dataset['image_license']}")

    if existing >= dataset["raw_pool_images"] and not args.force:
        print(f"  images      : {existing} already in {paths['raw_images']} (use --force to re-download)")
    else:
        with tempfile.TemporaryDirectory(prefix="annotation-qc-") as temp_dir:
            archive_path = Path(temp_dir) / "coco128.zip"
            download_archive(dataset["zip_url"], archive_path)
            checksum = sha256_file(archive_path)
            if checksum != dataset["zip_sha256"]:
                print(f"  ERROR: sha256 mismatch for downloaded archive")
                print(f"    expected: {dataset['zip_sha256']}")
                print(f"    actual  : {checksum}")
                return 1
            extracted = extract_members(archive_path, IMAGE_MEMBER_PREFIX, raw_images, ".jpg")
            print(f"  archive     : {human_size(archive_path.stat().st_size)} (sha256 verified)")
            print(f"  images      : {extracted} extracted to {paths['raw_images']}")
            if args.with_reference:
                reference_dir = root / paths["reference_labels"]
                labels = extract_members(archive_path, LABEL_MEMBER_PREFIX, reference_dir, ".txt")
                print(f"  reference   : {labels} COCO label files in {paths['reference_labels']}")

    image_count = len(list(raw_images.glob("*.jpg")))
    print(f"  raw pool    : {image_count} images ready")
    if not args.with_reference:
        print("  note        : COCO reference labels not extracted (add --with-reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
