"""Agreement check against the original COCO reference labels.

The reference labels shipped with coco128 are used **only** as an external
signal. They are not ground truth: the reference may miss small or occluded
objects, and the manual annotations may be the more correct ones. Differences
are reported so a human can review them, never auto-corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from annotation_qc.schemas import Annotation

DEFAULT_IOU_THRESHOLD = 0.5

# Class index in the standard COCO 80-class ordering used by the coco128
# label files (person, bicycle, car, ... dog at index 16).
COCO_INDEX_BY_CLASS = {"person": 0, "bicycle": 1, "car": 2, "dog": 16}


@dataclass
class MatchResult:
    matched: list[tuple[Annotation, Annotation, float]] = field(default_factory=list)
    manual_only: list[Annotation] = field(default_factory=list)
    reference_only: list[Annotation] = field(default_factory=list)


def parse_reference_label_file(
    path: Path, image_id: int, image_width: int, image_height: int, config: dict
) -> list[Annotation]:
    """Parse one YOLO-format reference label file into target-class boxes."""
    class_id_by_name = {category["name"]: category["id"] for category in config["classes"]}
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        coco_index = int(float(parts[0]))
        name = next(
            (candidate for candidate, index in COCO_INDEX_BY_CLASS.items() if index == coco_index),
            None,
        )
        if name is None or name not in class_id_by_name:
            continue
        center_x, center_y, width, height = (float(value) for value in parts[1:])
        x_min = (center_x - width / 2) * image_width
        y_min = (center_y - height / 2) * image_height
        boxes.append(
            Annotation(
                annotation_id=-1,
                image_id=image_id,
                category_id=class_id_by_name[name],
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + width * image_width,
                y_max=y_min + height * image_height,
            )
        )
    return boxes


def match_annotations(
    manual: list[Annotation], reference: list[Annotation], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> MatchResult:
    """Greedy one-to-one matching of same-class boxes by IoU."""
    result = MatchResult()
    available = list(manual)
    for reference_box in reference:
        best = None
        best_iou = 0.0
        for manual_box in available:
            if manual_box.category_id != reference_box.category_id:
                continue
            overlap = manual_box.iou(reference_box)
            if overlap >= iou_threshold and overlap > best_iou:
                best = manual_box
                best_iou = overlap
        if best is None:
            result.reference_only.append(reference_box)
        else:
            result.matched.append((best, reference_box, best_iou))
            available.remove(best)
    result.manual_only = available
    return result
