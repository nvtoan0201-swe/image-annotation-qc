"""Dataset statistics helpers.

All numbers are computed from the actual data on disk; nothing is hard-coded
or estimated.
"""

from __future__ import annotations

from collections import Counter

from annotation_qc.schemas import Annotation, AnnotationFile, ImageRecord


def percentile(values: list[float], percent: float) -> float:
    """Linear-interpolation percentile (percent in [0, 100])."""
    if not values:
        raise ValueError("percentile of an empty list is undefined")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def counts_by_image(annotation_file: AnnotationFile, image_ids: list[int]) -> dict[int, int]:
    """Number of annotations per image id (zero for images without boxes)."""
    counts = Counter(annotation.image_id for annotation in annotation_file.annotations)
    return {image_id: counts.get(image_id, 0) for image_id in image_ids}


def counts_by_class(annotations: list[Annotation], annotation_file: AnnotationFile) -> dict[str, int]:
    """Number of annotations per class name, sorted by count (descending)."""
    counter: Counter[str] = Counter()
    for annotation in annotations:
        name = annotation_file.category_name(annotation.category_id)
        counter[name or f"category_id:{annotation.category_id}"] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": round(min(values), 2),
        "p10": round(percentile(values, 10), 2),
        "median": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "max": round(max(values), 2),
    }


def build_dataset_report(
    annotation_file: AnnotationFile,
    manifest: list[ImageRecord],
) -> dict:
    """Compute the dataset report from real annotation and manifest data."""
    image_ids = [record.image_id for record in manifest]
    counts = counts_by_image(annotation_file, image_ids)
    annotations = annotation_file.annotations
    per_class = counts_by_class(annotations, annotation_file)
    total = len(annotations)

    images_per_class: dict[str, int] = {}
    for annotation in annotations:
        name = annotation_file.category_name(annotation.category_id) or f"category_id:{annotation.category_id}"
        images_per_class.setdefault(name, set())
        images_per_class[name].add(annotation.image_id)
    images_per_class_counts = {
        name: len(ids) for name, ids in sorted(images_per_class.items(), key=lambda item: -len(item[1]))
    }

    dimensions = [
        (record.width, record.height)
        for record in manifest
        if record.width and record.height
    ]
    box_widths = [annotation.width for annotation in annotations if annotation.area > 0]
    box_heights = [annotation.height for annotation in annotations if annotation.area > 0]
    box_areas = [annotation.area for annotation in annotations if annotation.area > 0]

    class_counts = list(per_class.values())
    imbalance = round(max(class_counts) / min(class_counts), 2) if class_counts and min(class_counts) > 0 else None

    return {
        "images": {
            "total": len(manifest),
            "annotated": sum(1 for count in counts.values() if count > 0),
            "without_annotations": sum(1 for count in counts.values() if count == 0),
        },
        "annotations": {
            "total": total,
            "per_class": per_class,
            "per_class_share": {
                name: round(count / total, 4) if total else 0.0 for name, count in per_class.items()
            },
            "class_imbalance_ratio": imbalance,
        },
        "images_per_class": images_per_class_counts,
        "annotations_per_image": {
            **_summary([float(count) for count in counts.values()]),
            "total": total,
        },
        "image_dimensions": {
            "width": _summary([float(width) for width, _ in dimensions]),
            "height": _summary([float(height) for _, height in dimensions]),
            "megapixels_median": round(
                percentile([width * height / 1_000_000 for width, height in dimensions], 50), 2
            )
            if dimensions
            else None,
        },
        "box_size": {
            "width": _summary(box_widths),
            "height": _summary(box_heights),
            "area": _summary(box_areas),
        },
    }
