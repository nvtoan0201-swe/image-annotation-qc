"""Tests for image and annotation validators."""

from __future__ import annotations

from helpers import (
    annotation_file,
    findings_by_check,
    make_config,
    make_manifest,
    write_image,
)

import pytest

from annotation_qc.schemas import Annotation
from annotation_qc.validators import ERROR, WARNING, validate_annotations, validate_image_files


@pytest.fixture
def config():
    return make_config()


def test_valid_annotations_produce_no_findings(config):
    manifest = make_manifest()
    boxes = [
        (1, 1, 0, (5, 5, 30, 60)),
        (2, 2, 1, (10, 10, 40, 30)),
        (3, 3, 2, (0, 0, 20, 20)),
        (4, 3, 3, (50, 30, 25, 25)),
    ]
    findings = validate_annotations(annotation_file(boxes=boxes), manifest, config)
    assert findings == []


def test_zero_width_box_is_invalid(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (10, 10, 0, 20))]), make_manifest(), config
    )
    assert "invalid_box_size" in findings_by_check(findings, ERROR)


def test_zero_height_box_is_invalid(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (10, 10, 20, 0))]), make_manifest(), config
    )
    assert "invalid_box_size" in findings_by_check(findings, ERROR)


def test_negative_coordinates_are_invalid(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (-5, 10, 30, 30))]), make_manifest(), config
    )
    assert "negative_coordinates" in findings_by_check(findings, ERROR)


def test_out_of_bounds_coordinates_are_invalid(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (90, 10, 30, 30))]), make_manifest(), config
    )
    assert "out_of_bounds" in findings_by_check(findings, ERROR)


def test_unknown_class_is_invalid(config):
    file = annotation_file(
        categories={0: "person", 1: "car", 2: "dog", 3: "bicycle", 9: "cat"},
        boxes=[(1, 1, 9, (10, 10, 30, 30))],
    )
    findings = validate_annotations(file, make_manifest(), config)
    checks = findings_by_check(findings, ERROR)
    assert "unknown_class" in checks
    assert len(checks["unknown_class"]) == 2


def test_annotation_for_image_missing_from_file(config):
    file = annotation_file(image_ids=(1, 2), boxes=[(1, 1, 0, (10, 10, 20, 20))])
    file.annotations.append(
        Annotation(
            annotation_id=2, image_id=99, category_id=0, x_min=0, y_min=0, x_max=10, y_max=10
        )
    )
    findings = validate_annotations(file, make_manifest(image_ids=(1, 2)), config)
    assert "annotation_for_missing_image" in findings_by_check(findings, ERROR)


def test_image_outside_manifest_is_reported(config):
    manifest = make_manifest(image_ids=(1, 2))
    file = annotation_file(image_ids=(1, 2, 99), boxes=[(1, 99, 0, (10, 10, 20, 20))])
    findings = validate_annotations(file, manifest, config)
    assert "image_not_in_manifest" in findings_by_check(findings, WARNING)


def test_duplicate_annotations_are_reported(config):
    boxes = [
        (1, 1, 0, (10, 10, 30, 30)),
        (2, 1, 0, (10.1, 10.1, 30, 30)),
    ]
    findings = validate_annotations(annotation_file(boxes=boxes), make_manifest(), config)
    assert "duplicate_annotation" in findings_by_check(findings, WARNING)


def test_tiny_box_is_reported(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (10, 10, 3, 3))]), make_manifest(), config
    )
    assert "tiny_box" in findings_by_check(findings, WARNING)


def test_huge_box_is_reported(config):
    findings = validate_annotations(
        annotation_file(boxes=[(1, 1, 0, (1, 1, 98, 78))]), make_manifest(), config
    )
    assert "huge_box" in findings_by_check(findings, WARNING)


def test_image_without_annotations_is_reported(config):
    file = annotation_file(boxes=[(1, 1, 0, (10, 10, 30, 30))])
    findings = validate_annotations(file, make_manifest(), config)
    assert len(findings_by_check(findings, WARNING)["no_annotations"]) == 2


def test_empty_annotation_file_is_an_error(config):
    findings = validate_annotations(annotation_file(boxes=[]), make_manifest(), config)
    assert "empty_annotations" in findings_by_check(findings, ERROR)


def test_unusual_annotation_count_is_reported(config):
    manifest = make_manifest(image_ids=(1, 2, 3, 4, 5, 6))
    boxes = []
    annotation_id = 1
    for image_id, count in enumerate((1, 2, 3, 4, 5, 30), start=1):
        for index in range(count):
            boxes.append((annotation_id, image_id, 0, (2, 2 + index, 10, 5)))
            annotation_id += 1
    findings = validate_annotations(annotation_file(image_ids=(1, 2, 3, 4, 5, 6), boxes=boxes), manifest, config)
    assert "unusual_annotation_count" in findings_by_check(findings, WARNING)


def test_unused_class_is_reported(config):
    file = annotation_file(boxes=[(1, 1, 0, (10, 10, 30, 30))])
    findings = validate_annotations(file, make_manifest(), config)
    assert len(findings_by_check(findings, WARNING)["class_not_used"]) == 3


def test_validate_image_files_passes_for_valid_images(tmp_path):
    manifest = make_manifest(image_ids=(1, 2))
    for record in manifest:
        write_image(tmp_path / record.filename)
    findings = validate_image_files(manifest, tmp_path)
    assert [finding for finding in findings if finding.severity == ERROR] == []


def test_corrupted_image_is_reported(tmp_path):
    manifest = make_manifest(image_ids=(1,))
    (tmp_path / manifest[0].filename).write_text("this is not an image")
    findings = validate_image_files(manifest, tmp_path)
    assert "image_unreadable" in findings_by_check(findings, ERROR)


def test_missing_image_is_reported(tmp_path):
    findings = validate_image_files(make_manifest(image_ids=(1,)), tmp_path)
    assert "image_unreadable" in findings_by_check(findings, ERROR)


def test_duplicate_content_is_reported(tmp_path):
    manifest = make_manifest(image_ids=(1, 2))
    write_image(tmp_path / manifest[0].filename)
    write_image(tmp_path / manifest[1].filename)
    findings = validate_image_files(manifest, tmp_path)
    assert "duplicate_content" in findings_by_check(findings, WARNING)


def test_duplicate_filename_in_manifest_is_reported(tmp_path):
    manifest = make_manifest(image_ids=(1, 1))
    write_image(tmp_path / manifest[0].filename)
    findings = validate_image_files(manifest, tmp_path)
    assert "duplicate_filename" in findings_by_check(findings, ERROR)
