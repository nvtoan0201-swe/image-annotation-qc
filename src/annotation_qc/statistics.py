"""Dataset statistics helpers.

All numbers are computed from the actual data on disk; nothing is hard-coded
or estimated.
"""

from __future__ import annotations

from collections import Counter

from annotation_qc.schemas import Annotation, AnnotationFile


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
