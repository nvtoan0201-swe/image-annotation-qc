"""Model-assisted pre-annotation conversion helpers.

Pre-annotations are suggestions produced by a pretrained detector. They are
**not** labels: every suggested box must be reviewed, corrected or rejected by
a human, and missing objects must be added, before an export is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from annotation_qc.schemas import ImageRecord

FROM_NAME = "label"
TO_NAME = "image"


@dataclass(frozen=True)
class Detection:
    """One object detection in pixel coordinates."""

    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def detection_to_result(
    detection: Detection,
    image_width: int,
    image_height: int,
    result_id: str,
    from_name: str = FROM_NAME,
    to_name: str = TO_NAME,
) -> dict | None:
    """Convert a detection to a Label Studio rectangle prediction.

    Label Studio stores rectangle values as percentages of the image size.
    Boxes that fall completely outside the image are dropped.
    """
    x_min = min(max(detection.x_min, 0.0), float(image_width))
    y_min = min(max(detection.y_min, 0.0), float(image_height))
    x_max = min(max(detection.x_max, 0.0), float(image_width))
    y_max = min(max(detection.y_max, 0.0), float(image_height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "origin": "prediction",
        "value": {
            "x": round(x_min / image_width * 100, 4),
            "y": round(y_min / image_height * 100, 4),
            "width": round((x_max - x_min) / image_width * 100, 4),
            "height": round((y_max - y_min) / image_height * 100, 4),
            "rotation": 0,
            "rectanglelabels": [detection.class_name],
        },
        "score": round(detection.confidence, 4),
    }


def build_preannotated_task(
    record: ImageRecord,
    detections: list[Detection],
    url_prefix: str,
    model_version: str,
) -> dict:
    """Build one Label Studio task containing a pre-annotation prediction."""
    if not record.width or not record.height:
        raise ValueError(f"{record.filename}: manifest has no image dimensions")
    results = []
    scores = []
    for index, detection in enumerate(detections):
        result = detection_to_result(
            detection,
            record.width,
            record.height,
            result_id=f"pre-{record.image_id}-{index}",
        )
        if result is not None:
            results.append(result)
            scores.append(detection.confidence)
    task = {"id": record.image_id, "data": {"image": f"{url_prefix}{record.filename}"}}
    if results:
        task["predictions"] = [
            {
                "model_version": model_version,
                "score": round(sum(scores) / len(scores), 4),
                "result": results,
            }
        ]
    return task
