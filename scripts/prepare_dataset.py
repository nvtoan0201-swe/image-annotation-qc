#!/usr/bin/env python3
"""Build the dataset manifest and the Label Studio task file.

Assigns stable image ids (the numeric COCO file name), inspects every selected
image, detects duplicates, and writes:

  - data/processed/images.csv   dataset manifest
  - data/annotations/tasks.json Label Studio tasks referencing local files
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from annotation_qc.io import (
    find_project_root,
    inspect_image,
    load_config,
    read_selected_images,
    write_label_studio_tasks,
    write_manifest,
)
from annotation_qc.schemas import ImageRecord

LOCAL_FILE_URL_PREFIX = "/data/local-files/?d=raw/"


def build_records(image_dir, filenames, source: str, license_name: str) -> list[ImageRecord]:
    records = []
    for position, filename in enumerate(sorted(filenames), start=1):
        stem = filename.rsplit(".", 1)[0]
        try:
            image_id = int(stem)
        except ValueError:
            image_id = position
            note = "non-numeric file name, positional id assigned"
        else:
            note = ""
        info = inspect_image(image_dir / filename)
        if info.error:
            status = "missing" if info.error == "file not found" else "unreadable"
            records.append(
                ImageRecord(
                    image_id=image_id,
                    filename=filename,
                    source=source,
                    license=license_name,
                    status=status,
                    note=info.error,
                )
            )
        else:
            records.append(
                ImageRecord(
                    image_id=image_id,
                    filename=filename,
                    source=source,
                    license=license_name,
                    width=info.width,
                    height=info.height,
                    sha256=info.sha256,
                    status="ok",
                    note=note,
                )
            )
    return records


def mark_duplicates(records: list[ImageRecord]) -> int:
    by_checksum: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.is_ok and record.sha256:
            by_checksum[record.sha256].append(record)
    duplicates = 0
    for group in by_checksum.values():
        if len(group) < 2:
            continue
        keeper = group[0]
        for duplicate in group[1:]:
            duplicate.status = "duplicate"
            duplicate.note = f"identical content to {keeper.filename}"
            duplicates += 1
    return duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    dataset = config["dataset"]
    paths = config["paths"]
    image_dir = root / paths["raw_images"]

    if not image_dir.is_dir() or not any(image_dir.glob("*.jpg")):
        print(f"ERROR: no images in {paths['raw_images']}. Run scripts/download_data.py first.")
        return 1

    selected = read_selected_images(root / paths["selected_images"])
    if len(set(selected)) != len(selected):
        print("ERROR: configs/selected_images.txt contains duplicate entries")
        return 1

    records = build_records(
        image_dir, selected, source=dataset["name"], license_name=dataset["image_license"]
    )
    duplicates = mark_duplicates(records)
    records.sort(key=lambda record: record.image_id)

    status_counts = defaultdict(int)
    for record in records:
        status_counts[record.status] += 1

    manifest_path = root / paths["manifest"]
    write_manifest(manifest_path, records)
    tasks = write_label_studio_tasks(
        root / paths["tasks"], [r for r in records if r.is_ok], LOCAL_FILE_URL_PREFIX
    )

    raw_pool = len(list(image_dir.glob("*.jpg")))
    class_names = ", ".join(category["name"] for category in config["classes"])

    print("Dataset preparation")
    print(f"  raw pool     : {raw_pool} images in {paths['raw_images']}")
    print(f"  selected     : {len(records)} images")
    print(
        "  status       : "
        f"{status_counts['ok']} ok, {status_counts['missing']} missing, "
        f"{status_counts['unreadable']} unreadable, {duplicates} duplicate"
    )
    print(f"  classes      : {class_names}")
    print(f"  manifest     : {paths['manifest']}")
    print(f"  tasks        : {paths['tasks']} ({tasks} tasks)")

    if status_counts["missing"] or status_counts["unreadable"]:
        print("  ERROR       : some selected images could not be used; fix before annotating")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
