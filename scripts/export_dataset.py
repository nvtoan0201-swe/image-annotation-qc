#!/usr/bin/env python3
"""Export the manually annotated dataset to YOLO format.

Runs the annotation QC validation first and refuses to export when errors are
found. The output is written to data/processed/yolo/ (images/, labels/,
classes.txt, dataset.yaml, export_summary.json).
"""

from __future__ import annotations

import argparse

from annotation_qc.exporter import ExportRefused, export_yolo_dataset
from annotation_qc.io import (
    CocoFormatError,
    find_project_root,
    load_coco,
    load_config,
    read_manifest,
)

MAX_REPORTED_ERRORS = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None, help="override the annotation file path")
    parser.add_argument("--output", default=None, help="override the export directory")
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]
    annotation_path = root / (args.annotations or paths["annotations"])
    output_dir = root / (args.output or paths["export"])

    print("Dataset export (YOLO)")
    if not annotation_path.is_file():
        print("  MANUAL STEP NOT COMPLETE")
        print(f"  expected annotation file: {paths['annotations']}")
        return 1

    manifest = read_manifest(root / paths["manifest"])
    try:
        annotation_file = load_coco(annotation_path)
    except CocoFormatError as error:
        print(f"  ERROR: {error}")
        return 2

    try:
        result = export_yolo_dataset(
            annotation_file, manifest, config, root / paths["raw_images"], output_dir
        )
    except ExportRefused as refused:
        print(f"  EXPORT REFUSED: {len(refused.findings)} validation errors must be fixed first")
        for finding in refused.findings[:MAX_REPORTED_ERRORS]:
            location = finding.filename or (
                f"image {finding.image_id}" if finding.image_id else "dataset"
            )
            if finding.annotation_id is not None:
                location += f"/annotation {finding.annotation_id}"
            print(f"    ERROR [{finding.check}] {location}: {finding.message}")
        if len(refused.findings) > MAX_REPORTED_ERRORS:
            print(f"    ... and {len(refused.findings) - MAX_REPORTED_ERRORS} more errors")
        print("  run: uv run scripts/validate_annotations.py")
        return 1

    print(f"  validation      : {result.errors} errors, {result.warnings} warnings (warnings do not block)")
    print(f"  images          : {result.images} copied to {output_dir.relative_to(root)}/images")
    print(f"  labels          : {result.labels} bounding boxes in {result.images} label files")
    print(f"  classes         : " + ", ".join(
        f"{index}={name}" for index, name in enumerate(result.classes)
    ))
    print(f"  export          : {output_dir.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
