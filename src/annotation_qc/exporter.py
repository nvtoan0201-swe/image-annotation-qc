"""Export the annotated dataset to YOLO format.

The export refuses to run when annotation validation reports errors; warnings
do not block the export but are part of the result.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from annotation_qc.io import write_json
from annotation_qc.schemas import Annotation, AnnotationFile, ImageRecord
from annotation_qc.validators import (
    ERROR,
    WARNING,
    Finding,
    count_by_severity,
    validate_annotations,
)


class ExportRefused(RuntimeError):
    """Raised when the annotation file has validation errors."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__(f"export refused: {len(findings)} validation errors")


@dataclass
class ExportResult:
    output_dir: Path
    images: int
    labels: int
    classes: list[str]
    errors: int
    warnings: int


def format_yolo_line(
    class_id: int, annotation: Annotation, image_width: int, image_height: int
) -> str:
    """One YOLO label line: class_id cx cy w h, all normalized to [0, 1]."""
    center_x = (annotation.x_min + annotation.x_max) / 2 / image_width
    center_y = (annotation.y_min + annotation.y_max) / 2 / image_height
    width = annotation.width / image_width
    height = annotation.height / image_height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def export_yolo_dataset(
    annotation_file: AnnotationFile,
    manifest: list[ImageRecord],
    config: dict,
    raw_images_dir: Path,
    output_dir: Path,
) -> ExportResult:
    """Validate, then write images/, labels/, classes.txt and dataset.yaml."""
    findings = validate_annotations(annotation_file, manifest, config)
    counts = count_by_severity(findings)
    if counts[ERROR]:
        raise ExportRefused([finding for finding in findings if finding.severity == ERROR])

    sorted_classes = sorted(config["classes"], key=lambda category: category["id"])
    class_ids = {category["name"]: category["id"] for category in sorted_classes}
    class_names = [category["name"] for category in sorted_classes]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    annotations_by_image: dict[int, list[Annotation]] = {}
    for annotation in annotation_file.annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    written_images = 0
    written_labels = 0
    for record in sorted(manifest, key=lambda item: item.image_id):
        if not record.is_ok:
            continue
        if not record.width or not record.height:
            raise ValueError(f"{record.filename}: manifest has no image dimensions")
        source = raw_images_dir / record.filename
        if not source.is_file():
            raise FileNotFoundError(f"{source}: image listed in the manifest is missing")
        shutil.copyfile(source, images_dir / record.filename)
        lines = []
        for annotation in annotations_by_image.get(record.image_id, []):
            name = annotation_file.category_name(annotation.category_id)
            lines.append(format_yolo_line(class_ids[name], annotation, record.width, record.height))
        written_labels += len(lines)
        (labels_dir / f"{record.filename.rsplit('.', 1)[0]}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        written_images += 1

    (output_dir / "classes.txt").write_text("".join(f"{name}\n" for name in class_names), encoding="utf-8")
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images",
                "val": "images",
                "names": {category["id"]: category["name"] for category in sorted_classes},
            },
            handle,
            sort_keys=False,
        )
    write_json(
        output_dir / "export_summary.json",
        {
            "images": written_images,
            "labels": written_labels,
            "classes": class_names,
            "errors": counts[ERROR],
            "warnings": counts[WARNING],
        },
    )
    return ExportResult(
        output_dir=output_dir,
        images=written_images,
        labels=written_labels,
        classes=class_names,
        errors=counts[ERROR],
        warnings=counts[WARNING],
    )
