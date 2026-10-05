"""Tests for dataset statistics."""

from __future__ import annotations

from helpers import annotation_file, make_manifest

import pytest

from annotation_qc.schemas import Annotation
from annotation_qc.statistics import (
    build_dataset_report,
    counts_by_class,
    counts_by_image,
    percentile,
)


def test_percentile_endpoints_and_middle():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 0.0
    assert percentile(values, 50) == 2.0
    assert percentile(values, 100) == 4.0


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([0.0, 1.0, 2.0, 3.0, 4.0], 90) == pytest.approx(3.6)


def test_percentile_single_value():
    assert percentile([7.0], 25) == 7.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_counts_by_image_includes_zero_counts():
    file = annotation_file(boxes=[(1, 1, 0, (10, 10, 20, 20)), (2, 3, 1, (10, 10, 20, 20))])
    assert counts_by_image(file, [1, 2, 3]) == {1: 1, 2: 0, 3: 1}


def test_counts_by_class_is_sorted_by_count():
    file = annotation_file(
        boxes=[
            (1, 1, 0, (10, 10, 20, 20)),
            (2, 1, 1, (40, 10, 20, 20)),
            (3, 1, 1, (60, 10, 20, 20)),
        ]
    )
    assert counts_by_class(file.annotations, file) == {"car": 2, "person": 1}


def test_dataset_report_totals_and_distributions():
    manifest = make_manifest(image_ids=(1, 2), width=200, height=100)
    boxes = [
        (1, 1, 0, (10, 10, 40, 60)),
        (2, 1, 0, (60, 10, 40, 60)),
        (3, 2, 1, (10, 10, 80, 40)),
    ]
    report = build_dataset_report(annotation_file(image_ids=(1, 2), boxes=boxes), manifest)

    assert report["images"] == {"total": 2, "annotated": 2, "without_annotations": 0}
    assert report["annotations"]["total"] == 3
    assert report["annotations"]["per_class"] == {"person": 2, "car": 1}
    assert report["annotations"]["per_class_share"]["person"] == pytest.approx(0.6667)
    assert report["annotations"]["class_imbalance_ratio"] == 2.0
    assert report["images_per_class"] == {"person": 1, "car": 1}
    assert report["annotations_per_image"]["max"] == 2.0
    assert report["annotations_per_image"]["min"] == 1.0
    assert report["image_dimensions"]["width"]["median"] == 200.0
    assert report["image_dimensions"]["height"]["median"] == 100.0
    assert report["box_size"]["width"]["median"] == 40.0
    assert report["box_size"]["height"]["median"] == 60.0


def test_dataset_report_with_empty_annotations():
    manifest = make_manifest(image_ids=(1, 2))
    report = build_dataset_report(annotation_file(image_ids=(1, 2), boxes=[]), manifest)

    assert report["annotations"]["total"] == 0
    assert report["annotations"]["per_class"] == {}
    assert report["annotations"]["class_imbalance_ratio"] is None
    assert report["images_per_class"] == {}
    assert report["box_size"]["area"] is None
    assert report["annotations_per_image"]["total"] == 0
    assert report["images"]["without_annotations"] == 2


def test_dataset_report_ignores_invalid_boxes_for_size_stats():
    manifest = make_manifest(image_ids=(1,))
    file = annotation_file(image_ids=(1,))
    file.annotations.append(
        Annotation(annotation_id=1, image_id=1, category_id=0, x_min=5, y_min=5, x_max=5, y_max=25)
    )
    report = build_dataset_report(file, manifest)
    assert report["annotations"]["total"] == 1
    assert report["box_size"]["width"] is None
