#!/usr/bin/env python3
"""Quality-control check for the manually created annotations.

Reads the Label Studio COCO export, compares it with the dataset manifest and
the labeling guidelines, and writes a report. Nothing is modified or deleted:
each finding must be reviewed by a human.

If the annotation file does not exist yet, the MANUAL annotation step has not
been completed (see README, section "Annotation").
"""

from __future__ import annotations

import argparse

from annotation_qc.io import (
    CocoFormatError,
    find_project_root,
    load_coco,
    load_config,
    read_manifest,
    write_json,
)
from annotation_qc.statistics import counts_by_class
from annotation_qc.validators import ERROR, WARNING, Finding, count_by_severity, validate_annotations

MAX_FINDINGS_PER_SEVERITY = 40


def render_markdown(report: dict) -> str:
    lines = [
        "# Annotation QC Report",
        "",
        f"Annotation file: `{report['annotation_file']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Images in manifest | {report['manifest_images']} |",
        f"| Images with annotations | {report['images_with_annotations']} |",
        f"| Images without annotations | {report['images_without_annotations']} |",
        f"| Total annotations | {report['total_annotations']} |",
        f"| Errors | {report['errors']} |",
        f"| Warnings | {report['warnings']} |",
        "",
        "## Annotations per class",
        "",
        "| Class | Count |",
        "| --- | --- |",
    ]
    for name, count in report["annotations_per_class"].items():
        lines.append(f"| {name} | {count} |")
    lines.append("")

    for severity, title in ((ERROR, "Errors"), (WARNING, "Warnings")):
        selected = [finding for finding in report["findings"] if finding["severity"] == severity]
        lines.append(f"## {title} ({len(selected)})")
        lines.append("")
        if not selected:
            lines.append("None.")
        for finding in selected:
            location = finding.get("filename") or (
                f"image {finding['image_id']}" if finding.get("image_id") else "dataset"
            )
            annotation = (
                f"/annotation {finding['annotation_id']}" if finding.get("annotation_id") else ""
            )
            lines.append(f"- **{finding['check']}** ({location}{annotation}): {finding['message']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def print_findings(findings: list[Finding]) -> None:
    for severity in (ERROR, WARNING):
        selected = [finding for finding in findings if finding.severity == severity]
        for finding in selected[:MAX_FINDINGS_PER_SEVERITY]:
            prefix = "ERROR  " if severity == ERROR else "WARNING"
            location = finding.filename or (
                f"image {finding.image_id}" if finding.image_id else "dataset"
            )
            if finding.annotation_id is not None:
                location += f"/annotation {finding.annotation_id}"
            print(f"    {prefix} [{finding.check}] {location}: {finding.message}")
        remaining = len(selected) - MAX_FINDINGS_PER_SEVERITY
        if remaining > 0:
            print(f"    ... and {remaining} more {severity}s (see the report)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        default=None,
        help="COCO JSON or Label Studio export .zip (default: path from configs/classes.yaml)",
    )
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]
    annotation_path = root / (args.annotations or paths["annotations"])

    print("Annotation QC")
    if not annotation_path.is_file():
        print("  MANUAL STEP NOT COMPLETE")
        print(f"  expected annotation file: {paths['annotations']}")
        print("  annotate the images in Label Studio, export as COCO, and place the JSON there")
        print("  workflow: see README.md")
        return 1

    manifest = read_manifest(root / paths["manifest"])
    try:
        annotation_file = load_coco(annotation_path)
    except CocoFormatError as error:
        print(f"  ERROR: {error}")
        return 2

    findings = validate_annotations(annotation_file, manifest, config)
    counts = count_by_severity(findings)
    per_class = counts_by_class(annotation_file.annotations, annotation_file)
    annotated_ids = {annotation.image_id for annotation in annotation_file.annotations}

    report = {
        "annotation_file": str(annotation_path.relative_to(root))
        if annotation_path.is_relative_to(root)
        else str(annotation_path),
        "manifest_images": len(manifest),
        "images_with_annotations": len(annotated_ids),
        "images_without_annotations": len(manifest) - len(annotated_ids),
        "total_annotations": len(annotation_file.annotations),
        "annotations_per_class": per_class,
        "errors": counts[ERROR],
        "warnings": counts[WARNING],
        "findings": [finding.to_dict() for finding in findings],
    }
    report_json = root / paths["reports"] / "annotation_qc.json"
    report_md = root / paths["reports"] / "annotation_qc.md"
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"  annotation file : {report['annotation_file']}")
    print(f"  manifest images : {len(manifest)}")
    print(
        f"  annotated       : {len(annotated_ids)} images "
        f"({report['images_without_annotations']} without annotations)"
    )
    print(f"  annotations     : {len(annotation_file.annotations)}")
    if per_class:
        print("  per class       : " + ", ".join(f"{name}={count}" for name, count in per_class.items()))
    print(f"  errors          : {counts[ERROR]}")
    print(f"  warnings        : {counts[WARNING]}")
    print_findings(findings)
    print(f"  reports         : {report_json.relative_to(root)}, {report_md.relative_to(root)}")
    print(f"  result          : {'FAIL' if counts[ERROR] else 'PASS'}")
    return 1 if counts[ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
