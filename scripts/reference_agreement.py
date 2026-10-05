#!/usr/bin/env python3
"""Compare manual annotations with the original COCO reference labels.

This is a quality-control SIGNAL, not ground truth. The reference labels may
miss objects and may disagree with the guidelines; every difference must be
reviewed by a human. Run `download_data.py --with-reference` first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from annotation_qc.agreement import DEFAULT_IOU_THRESHOLD, match_annotations, parse_reference_label_file
from annotation_qc.io import (
    CocoFormatError,
    find_project_root,
    load_coco,
    load_config,
    read_manifest,
    write_json,
)
from annotation_qc.statistics import percentile

UNKNOWN_CLASS = "category_id:{id}"


def empty_counters() -> dict:
    return {
        "manual_boxes": 0,
        "reference_boxes": 0,
        "matched": 0,
        "manual_only": 0,
        "reference_only": 0,
        "matched_iou_values": [],
    }


def finalize(counters: dict) -> dict:
    result = {key: value for key, value in counters.items() if key != "matched_iou_values"}
    ious = counters["matched_iou_values"]
    result["median_matched_iou"] = round(percentile(ious, 50), 3) if ious else None
    result["manual_match_rate"] = (
        round(result["matched"] / result["manual_boxes"], 3) if result["manual_boxes"] else None
    )
    result["reference_match_rate"] = (
        round(result["matched"] / result["reference_boxes"], 3) if result["reference_boxes"] else None
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None, help="override the annotation file path")
    parser.add_argument("--reference-labels", default=None, help="override the reference labels directory")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU match threshold")
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]
    annotation_path = root / (args.annotations or paths["annotations"])
    labels_dir = root / (args.reference_labels or paths["reference_labels"])

    print("Reference agreement (signal only, not ground truth)")
    if not annotation_path.is_file():
        print("  MANUAL STEP NOT COMPLETE")
        print(f"  expected annotation file: {paths['annotations']}")
        return 1
    if not labels_dir.is_dir() or not any(labels_dir.glob("*.txt")):
        print("  reference labels not found")
        print("  run: uv run scripts/download_data.py --with-reference")
        return 1

    manifest = read_manifest(root / paths["manifest"])
    try:
        annotation_file = load_coco(annotation_path)
    except CocoFormatError as error:
        print(f"  ERROR: {error}")
        return 2

    per_class: dict[str, dict] = defaultdict(empty_counters)
    overall = empty_counters()
    compared = 0
    for record in sorted(manifest, key=lambda item: item.image_id):
        if not record.is_ok or not record.width or not record.height:
            continue
        reference_boxes = parse_reference_label_file(
            labels_dir / f"{record.filename.rsplit('.', 1)[0]}.txt",
            record.image_id,
            record.width,
            record.height,
            config,
        )
        manual_boxes = annotation_file.annotations_for(record.image_id)
        if not reference_boxes and not manual_boxes:
            continue
        compared += 1
        result = match_annotations(manual_boxes, reference_boxes, args.iou)
        for manual_box in manual_boxes:
            name = annotation_file.category_name(manual_box.category_id) or UNKNOWN_CLASS.format(
                id=manual_box.category_id
            )
            per_class[name]["manual_boxes"] += 1
            overall["manual_boxes"] += 1
        for reference_box in reference_boxes:
            name = annotation_file.category_name(reference_box.category_id) or UNKNOWN_CLASS.format(
                id=reference_box.category_id
            )
            per_class[name]["reference_boxes"] += 1
            overall["reference_boxes"] += 1
        for manual_box, reference_box, overlap in result.matched:
            name = annotation_file.category_name(manual_box.category_id) or UNKNOWN_CLASS.format(
                id=manual_box.category_id
            )
            per_class[name]["matched"] += 1
            per_class[name]["matched_iou_values"].append(overlap)
            overall["matched"] += 1
            overall["matched_iou_values"].append(overlap)
        for manual_box in result.manual_only:
            name = annotation_file.category_name(manual_box.category_id) or UNKNOWN_CLASS.format(
                id=manual_box.category_id
            )
            per_class[name]["manual_only"] += 1
            overall["manual_only"] += 1
        for reference_box in result.reference_only:
            name = annotation_file.category_name(reference_box.category_id) or UNKNOWN_CLASS.format(
                id=reference_box.category_id
            )
            per_class[name]["reference_only"] += 1
            overall["reference_only"] += 1

    report = {
        "note": (
            "The COCO reference labels are an external signal, not ground truth. "
            "manual_only can be a correct correction or an over-label; "
            "reference_only can be a correct skip or a missed object."
        ),
        "iou_threshold": args.iou,
        "images_compared": compared,
        "overall": finalize(overall),
        "per_class": {name: finalize(counters) for name, counters in per_class.items()},
    }
    report_path = root / paths["reports"] / "reference_agreement.json"
    write_json(report_path, report)

    summary = report["overall"]
    print(f"  images compared : {compared}")
    print(f"  manual boxes    : {summary['manual_boxes']}")
    print(f"  reference boxes : {summary['reference_boxes']} (target classes only)")
    print(f"  matched         : {summary['matched']} (IoU >= {args.iou})")
    print(f"  manual only     : {summary['manual_only']}")
    print(f"  reference only  : {summary['reference_only']}")
    for name, counters in sorted(report["per_class"].items()):
        print(
            f"    {name:9s} manual={counters['manual_boxes']:4d} reference={counters['reference_boxes']:4d} "
            f"matched={counters['matched']:4d} manual_only={counters['manual_only']:4d} "
            f"reference_only={counters['reference_only']:4d}"
        )
    print(f"  report          : {report_path.relative_to(root)}")
    print("  reminder        : reference labels are not ground truth; review differences by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
