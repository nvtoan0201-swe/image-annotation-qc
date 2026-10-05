"""Core data structures shared across the pipeline.

Bounding boxes are stored internally in pixel coordinates as
``(x_min, y_min, x_max, y_max)``. COCO files store ``[x, y, width, height]``
and are converted on load; YOLO stores normalized center coordinates and is
generated on export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Category:
    """An object class available for annotation."""

    id: int
    name: str


@dataclass
class ImageRecord:
    """One image in the dataset, as tracked in the dataset manifest."""

    image_id: int
    filename: str
    source: str = ""
    license: str = ""
    width: int | None = None
    height: int | None = None
    sha256: str = ""
    status: str = "unknown"
    note: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "width": self.width if self.width is not None else "",
            "height": self.height if self.height is not None else "",
            "source": self.source,
            "license": self.license,
            "sha256": self.sha256,
            "status": self.status,
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ImageRecord":
        width = int(row["width"]) if str(row.get("width", "")).strip() else None
        height = int(row["height"]) if str(row.get("height", "")).strip() else None
        return cls(
            image_id=int(row["image_id"]),
            filename=row["filename"],
            source=row.get("source", ""),
            license=row.get("license", ""),
            width=width,
            height=height,
            sha256=row.get("sha256", ""),
            status=row.get("status", "unknown"),
            note=row.get("note", ""),
        )


@dataclass
class Annotation:
    """One bounding box on one image."""

    annotation_id: int
    image_id: int
    category_id: int
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

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width / 2.0, self.y_min + self.height / 2.0)

    def iou(self, other: "Annotation") -> float:
        """Intersection over union with another box (0.0 for no overlap)."""
        inter_w = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        inter_h = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def from_coco(cls, item: dict[str, Any]) -> "Annotation":
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation {item.get('id')}: bbox must be [x, y, width, height]")
        x, y, width, height = (float(value) for value in bbox)
        return cls(
            annotation_id=int(item["id"]),
            image_id=int(item["image_id"]),
            category_id=int(item["category_id"]),
            x_min=x,
            y_min=y,
            x_max=x + width,
            y_max=y + height,
        )


@dataclass
class AnnotationFile:
    """A parsed annotation file: images, classes and bounding boxes."""

    images: dict[int, ImageRecord] = field(default_factory=dict)
    categories: dict[int, Category] = field(default_factory=dict)
    annotations: list[Annotation] = field(default_factory=list)

    def annotations_for(self, image_id: int) -> list[Annotation]:
        return [a for a in self.annotations if a.image_id == image_id]

    def category_name(self, category_id: int) -> str | None:
        category = self.categories.get(category_id)
        return category.name if category else None
