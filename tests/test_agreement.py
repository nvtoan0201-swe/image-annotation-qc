"""Tests for the reference-label agreement check."""

from __future__ import annotations

import pytest

from helpers import make_config

from annotation_qc.agreement import match_annotations, parse_reference_label_file
from annotation_qc.schemas import Annotation


def box(annotation_id: int, category_id: int, x: float, y: float, width: float, height: float) -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        image_id=1,
        category_id=category_id,
        x_min=x,
        y_min=y,
        x_max=x + width,
        y_max=y + height,
    )


def test_match_exact_boxes():
    manual = [box(1, 0, 10, 10, 20, 20)]
    reference = [box(-1, 0, 10, 10, 20, 20)]
    result = match_annotations(manual, reference)
    assert len(result.matched) == 1
    assert result.matched[0][2] == pytest.approx(1.0)
    assert result.manual_only == []
    assert result.reference_only == []


def test_different_classes_do_not_match():
    manual = [box(1, 0, 10, 10, 20, 20)]
    reference = [box(-1, 1, 10, 10, 20, 20)]
    result = match_annotations(manual, reference)
    assert result.matched == []
    assert len(result.manual_only) == 1
    assert len(result.reference_only) == 1


def test_non_overlapping_boxes_do_not_match():
    manual = [box(1, 0, 0, 0, 10, 10)]
    reference = [box(-1, 0, 50, 50, 10, 10)]
    result = match_annotations(manual, reference)
    assert result.matched == []


def test_greedy_matching_uses_each_manual_box_once():
    manual = [box(1, 0, 10, 10, 20, 20)]
    reference = [box(-1, 0, 10, 10, 20, 20), box(-2, 0, 12, 12, 20, 20)]
    result = match_annotations(manual, reference)
    assert len(result.matched) == 1
    assert len(result.reference_only) == 1


def test_parse_reference_label_file_converts_coordinates(tmp_path):
    path = tmp_path / "000000000001.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n16 0.25 0.25 0.1 0.1\n45 0.5 0.5 0.5 0.5\n\nbroken\n")
    boxes = parse_reference_label_file(path, image_id=1, image_width=100, image_height=80, config=make_config())
    assert len(boxes) == 2
    person, dog = boxes
    assert person.category_id == 0
    assert person.x_min == pytest.approx(40)
    assert person.y_min == pytest.approx(32)
    assert person.x_max == pytest.approx(60)
    assert person.y_max == pytest.approx(48)
    assert dog.category_id == 2


def test_parse_reference_label_file_missing_file(tmp_path):
    boxes = parse_reference_label_file(
        tmp_path / "nope.txt", image_id=1, image_width=100, image_height=80, config=make_config()
    )
    assert boxes == []
