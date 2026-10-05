"""Automated checks for images and annotations.

Validators only report findings. They never modify or delete data: every
finding is meant to be reviewed by a human, who decides what to fix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from annotation_qc.io import inspect_image
from annotation_qc.schemas import Annotation, AnnotationFile, ImageRecord
from annotation_qc.statistics import counts_by_image, percentile

ERROR = "error"
WARNING = "warning"

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BOUNDARY_TOLERANCE_PX = 0.5
OUTLIER_IQR_MULTIPLIER = 1.5
MIN_IMAGES_FOR_OUTLIER_CHECK = 5


@dataclass
class Finding:
    """One issue (or informational note) produced by a validator."""

    check: str
    severity: str
    message: str
    image_id: int | None = None
    annotation_id: int | None = None
    filename: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    counts = {ERROR: 0, WARNING: 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def validate_image_files(records: list[ImageRecord], image_dir: Path) -> list[Finding]:
    """Validate that every manifest entry maps to a readable, unique image."""
    findings: list[Finding] = []

    by_name: dict[str, list[ImageRecord]] = defaultdict(list)
    by_checksum: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_name[record.filename].append(record)

    for filename, group in sorted(by_name.items()):
        if len(group) > 1:
            ids = ", ".join(str(record.image_id) for record in group)
            findings.append(
                Finding(
                    check="duplicate_filename",
                    severity=ERROR,
                    message=f"file name appears {len(group)} times in the manifest (image ids: {ids})",
                    filename=filename,
                )
            )

    for record in sorted(records, key=lambda item: item.image_id):
        path = image_dir / record.filename
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            findings.append(
                Finding(
                    check="unexpected_format",
                    severity=WARNING,
                    message=f"unexpected file extension '{suffix}'",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        info = inspect_image(path)
        if info.error:
            findings.append(
                Finding(
                    check="image_unreadable",
                    severity=ERROR,
                    message=info.error,
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
            continue
        if not info.width or not info.height or info.width <= 0 or info.height <= 0:
            findings.append(
                Finding(
                    check="invalid_dimensions",
                    severity=ERROR,
                    message=f"invalid image dimensions {info.width}x{info.height}",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
            continue
        if record.width and record.height and (info.width, info.height) != (record.width, record.height):
            findings.append(
                Finding(
                    check="dimensions_changed",
                    severity=WARNING,
                    message=(
                        f"manifest says {record.width}x{record.height}, file is "
                        f"{info.width}x{info.height}; re-run prepare_dataset.py"
                    ),
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        if record.sha256 and info.sha256 != record.sha256:
            findings.append(
                Finding(
                    check="content_changed",
                    severity=WARNING,
                    message="file content differs from the manifest checksum",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        by_checksum[info.sha256].append(record)

    for checksum, group in sorted(by_checksum.items()):
        if len(group) > 1:
            names = ", ".join(record.filename for record in group)
            findings.append(
                Finding(
                    check="duplicate_content",
                    severity=WARNING,
                    message=f"{len(group)} images share identical content: {names}",
                    image_id=group[0].image_id,
                )
            )

    return findings


def validate_annotations(
    annotation_file: AnnotationFile,
    manifest: list[ImageRecord],
    config: dict,
) -> list[Finding]:
    """Check an annotation file against the dataset manifest and the guidelines."""
    findings: list[Finding] = []
    thresholds = config["quality_control"]
    configured_names = {category["name"] for category in config["classes"]}
    manifest_by_id = {record.image_id: record for record in manifest}

    if not annotation_file.annotations:
        findings.append(
            Finding(
                check="empty_annotations",
                severity=ERROR,
                message="the annotation file contains no annotations at all",
            )
        )

    for image_id, image in sorted(annotation_file.images.items()):
        if image_id not in manifest_by_id:
            findings.append(
                Finding(
                    check="image_not_in_manifest",
                    severity=WARNING,
                    message=f"annotated image {image.filename} is not part of the dataset manifest",
                    image_id=image_id,
                    filename=image.filename,
                )
            )

    known_category_ids = set()
    for category in annotation_file.categories.values():
        if category.name in configured_names:
            known_category_ids.add(category.id)
        else:
            findings.append(
                Finding(
                    check="unknown_class",
                    severity=ERROR,
                    message=f"class '{category.name}' is not one of the configured classes "
                    f"({', '.join(sorted(configured_names))})",
                )
            )

    image_sizes: dict[int, tuple[int, int]] = {
        image_id: (image.width, image.height)
        for image_id, image in annotation_file.images.items()
        if image.width and image.height
    }
    for record in manifest:
        if record.image_id not in image_sizes and record.width and record.height:
            image_sizes[record.image_id] = (record.width, record.height)

    for annotation in annotation_file.annotations:
        location = {"image_id": annotation.image_id, "annotation_id": annotation.annotation_id}
        if annotation.category_id not in annotation_file.categories:
            findings.append(
                Finding(
                    check="unknown_class",
                    severity=ERROR,
                    message=f"annotation refers to undefined category id {annotation.category_id}",
                    **location,
                )
            )
        elif annotation.category_id not in known_category_ids:
            name = annotation_file.category_name(annotation.category_id)
            findings.append(
                Finding(
                    check="unknown_class",
                    severity=ERROR,
                    message=f"annotation uses non-target class '{name}'",
                    **location,
                )
            )
        if annotation.image_id not in annotation_file.images:
            findings.append(
                Finding(
                    check="annotation_for_missing_image",
                    severity=ERROR,
                    message="annotation refers to an image that is not in the annotation file",
                    **location,
                )
            )

        if annotation.x_min >= annotation.x_max or annotation.y_min >= annotation.y_max:
            findings.append(
                Finding(
                    check="invalid_box_size",
                    severity=ERROR,
                    message=(
                        f"box has zero or negative size "
                        f"(x: {annotation.x_min:.1f} -> {annotation.x_max:.1f}, "
                        f"y: {annotation.y_min:.1f} -> {annotation.y_max:.1f})"
                    ),
                    **location,
                )
            )
            continue

        if annotation.x_min < -BOUNDARY_TOLERANCE_PX or annotation.y_min < -BOUNDARY_TOLERANCE_PX:
            findings.append(
                Finding(
                    check="negative_coordinates",
                    severity=ERROR,
                    message=(
                        f"box starts outside the image "
                        f"(x_min={annotation.x_min:.1f}, y_min={annotation.y_min:.1f})"
                    ),
                    **location,
                )
            )

        size = image_sizes.get(annotation.image_id)
        if size:
            width, height = size
            if (
                annotation.x_max > width + BOUNDARY_TOLERANCE_PX
                or annotation.y_max > height + BOUNDARY_TOLERANCE_PX
            ):
                findings.append(
                    Finding(
                        check="out_of_bounds",
                        severity=ERROR,
                        message=(
                            f"box extends beyond the {width}x{height} image "
                            f"(x_max={annotation.x_max:.1f}, y_max={annotation.y_max:.1f})"
                        ),
                        **location,
                    )
                )
            if annotation.area > thresholds["max_box_area_fraction"] * width * height:
                findings.append(
                    Finding(
                        check="huge_box",
                        severity=WARNING,
                        message=(
                            f"box covers {annotation.area / (width * height):.1%} of the image; "
                            "check that it tightly covers a single object"
                        ),
                        **location,
                    )
                )

        if (
            min(annotation.width, annotation.height) < thresholds["min_box_side_px"]
            or annotation.area < thresholds["min_box_area_px"]
        ):
            findings.append(
                Finding(
                    check="tiny_box",
                    severity=WARNING,
                    message=(
                        f"box is very small ({annotation.width:.1f}x{annotation.height:.1f} px, "
                        f"area {annotation.area:.1f} px2); verify the object or remove it"
                    ),
                    **location,
                )
            )

    findings.extend(_find_duplicates(annotation_file, thresholds["duplicate_iou_threshold"]))
    findings.extend(_find_unannotated_images(annotation_file, manifest))
    findings.extend(_find_count_outliers(annotation_file, manifest))
    findings.extend(_find_unused_classes(annotation_file, configured_names))

    return findings


def _find_duplicates(annotation_file: AnnotationFile, iou_threshold: float) -> list[Finding]:
    grouped: dict[tuple[int, int], list[Annotation]] = defaultdict(list)
    for annotation in annotation_file.annotations:
        grouped[(annotation.image_id, annotation.category_id)].append(annotation)

    findings = []
    for group in grouped.values():
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                overlap = first.iou(second)
                if overlap >= iou_threshold:
                    findings.append(
                        Finding(
                            check="duplicate_annotation",
                            severity=WARNING,
                            message=(
                                f"annotations {first.annotation_id} and {second.annotation_id} "
                                f"overlap with IoU {overlap:.2f}; possible duplicate box"
                            ),
                            image_id=first.image_id,
                            annotation_id=second.annotation_id,
                        )
                    )
    return findings


def _find_unannotated_images(
    annotation_file: AnnotationFile, manifest: list[ImageRecord]
) -> list[Finding]:
    annotated_ids = {annotation.image_id for annotation in annotation_file.annotations}
    findings = []
    for record in sorted(manifest, key=lambda item: item.image_id):
        if record.image_id not in annotated_ids:
            findings.append(
                Finding(
                    check="no_annotations",
                    severity=WARNING,
                    message="image has no annotations in the export; confirm that it contains no target object",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
    return findings


def _find_count_outliers(
    annotation_file: AnnotationFile, manifest: list[ImageRecord]
) -> list[Finding]:
    counts = counts_by_image(annotation_file, [record.image_id for record in manifest])
    values = [count for count in counts.values() if count > 0]
    if len(values) < MIN_IMAGES_FOR_OUTLIER_CHECK:
        return []
    lower_quartile = percentile(values, 25)
    upper_quartile = percentile(values, 75)
    interquartile_range = upper_quartile - lower_quartile
    if interquartile_range <= 0:
        return []
    limit = upper_quartile + OUTLIER_IQR_MULTIPLIER * interquartile_range
    findings = []
    for image_id, count in counts.items():
        if count > limit:
            findings.append(
                Finding(
                    check="unusual_annotation_count",
                    severity=WARNING,
                    message=(
                        f"image has {count} annotations, above the outlier limit of "
                        f"{limit:.1f} (Q3={upper_quartile:.0f}, IQR={interquartile_range:.0f}); "
                        "check for split/merged boxes"
                    ),
                    image_id=image_id,
                )
            )
    return findings


def _find_unused_classes(annotation_file: AnnotationFile, configured_names: set[str]) -> list[Finding]:
    used = {
        annotation_file.category_name(annotation.category_id)
        for annotation in annotation_file.annotations
    }
    findings = []
    for name in sorted(configured_names - used):
        findings.append(
            Finding(
                check="class_not_used",
                severity=WARNING,
                message=f"configured class '{name}' has no annotations in this export",
            )
        )
    return findings
