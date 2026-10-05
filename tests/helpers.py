"""Shared test fixtures for the annotation_qc test suite."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from annotation_qc.schemas import Annotation, AnnotationFile, Category, ImageRecord

CLASS_SPECS = [(0, "person"), (1, "car"), (2, "dog"), (3, "bicycle")]


def make_config() -> dict:
    return {
        "classes": [{"id": category_id, "name": name} for category_id, name in CLASS_SPECS],
        "quality_control": {
            "min_box_side_px": 4,
            "min_box_area_px": 16,
            "max_box_area_fraction": 0.8,
            "duplicate_iou_threshold": 0.95,
            "min_annotations_per_image": 1,
        },
    }


def make_manifest(image_ids: tuple[int, ...] = (1, 2, 3), width: int = 100, height: int = 80):
    return [
        ImageRecord(
            image_id=image_id,
            filename=f"{image_id:012d}.jpg",
            width=width,
            height=height,
            status="ok",
        )
        for image_id in image_ids
    ]


def annotation_file(
    image_ids: tuple[int, ...] = (1, 2, 3),
    categories: dict[int, str] | None = None,
    boxes: list[tuple[int, int, int, tuple[float, float, float, float]]] | None = None,
    width: int = 100,
    height: int = 80,
) -> AnnotationFile:
    """Build an AnnotationFile.

    ``boxes`` entries are ``(annotation_id, image_id, category_id, (x, y, w, h))``.
    """
    annotation_file = AnnotationFile()
    for image_id in image_ids:
        annotation_file.images[image_id] = ImageRecord(
            image_id=image_id,
            filename=f"{image_id:012d}.jpg",
            width=width,
            height=height,
            status="ok",
        )
    for category_id, name in (categories or dict(CLASS_SPECS)).items():
        annotation_file.categories[category_id] = Category(id=category_id, name=name)
    for annotation_id, image_id, category_id, (x, y, box_width, box_height) in boxes or []:
        annotation_file.annotations.append(
            Annotation(
                annotation_id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                x_min=x,
                y_min=y,
                x_max=x + box_width,
                y_max=y + box_height,
            )
        )
    return annotation_file


def write_image(path: Path, size: tuple[int, int] = (100, 80), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def findings_by_check(findings, severity: str | None = None) -> dict[str, list]:
    result: dict[str, list] = {}
    for finding in findings:
        if severity and finding.severity != severity:
            continue
        result.setdefault(finding.check, []).append(finding)
    return result
