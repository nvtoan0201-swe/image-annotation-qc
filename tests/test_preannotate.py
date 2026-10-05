"""Tests for model-assisted pre-annotation conversion."""

from __future__ import annotations

import pytest

from helpers import make_manifest

from annotation_qc.preannotate import Detection, build_preannotated_task, detection_to_result


def detection(**overrides) -> Detection:
    values = {
        "class_name": "person",
        "confidence": 0.9,
        "x_min": 10.0,
        "y_min": 20.0,
        "x_max": 60.0,
        "y_max": 80.0,
    }
    values.update(overrides)
    return Detection(**values)


def test_detection_to_result_uses_percentages():
    result = detection_to_result(detection(), image_width=100, image_height=100, result_id="pre-1")
    assert result["type"] == "rectanglelabels"
    assert result["from_name"] == "label"
    assert result["to_name"] == "image"
    assert result["origin"] == "prediction"
    assert result["value"] == {
        "x": 10.0,
        "y": 20.0,
        "width": 50.0,
        "height": 60.0,
        "rotation": 0,
        "rectanglelabels": ["person"],
    }
    assert result["score"] == 0.9


def test_detection_to_result_rounds_percentage_values():
    result = detection_to_result(detection(x_max=33.0), image_width=200, image_height=80, result_id="x")
    assert result["value"]["x"] == 5.0
    assert result["value"]["width"] == 11.5


def test_detection_to_result_clips_to_image_bounds():
    result = detection_to_result(
        detection(x_min=-10, y_min=-5, x_max=120, y_max=90),
        image_width=100,
        image_height=100,
        result_id="clip",
    )
    assert result["value"]["x"] == 0.0
    assert result["value"]["y"] == 0.0
    assert result["value"]["width"] == 100.0
    assert result["value"]["height"] == 90.0


def test_detection_fully_outside_is_dropped():
    assert detection_to_result(
        detection(x_min=200, x_max=300), image_width=100, image_height=100, result_id="outside"
    ) is None


def test_detection_with_zero_size_is_dropped():
    assert detection_to_result(
        detection(x_min=10, x_max=10), image_width=100, image_height=100, result_id="zero"
    ) is None


def test_build_preannotated_task_structure():
    record = make_manifest(image_ids=(7,))[0]
    record.filename = "000000000007.jpg"
    task = build_preannotated_task(
        record,
        [detection(), detection(class_name="car", confidence=0.5)],
        url_prefix="/data/local-files/?d=data/raw/images/",
        model_version="yolov8n-coco",
    )
    assert task["id"] == 7
    assert task["data"]["image"] == "/data/local-files/?d=data/raw/images/000000000007.jpg"
    prediction = task["predictions"][0]
    assert prediction["model_version"] == "yolov8n-coco"
    assert prediction["score"] == pytest.approx(0.7)
    assert len(prediction["result"]) == 2
    assert prediction["result"][0]["id"] == "pre-7-0"
    assert prediction["result"][1]["id"] == "pre-7-1"
    assert [result["value"]["rectanglelabels"][0] for result in prediction["result"]] == ["person", "car"]


def test_build_preannotated_task_without_detections_has_no_predictions():
    record = make_manifest(image_ids=(8,))[0]
    task = build_preannotated_task(record, [], "/prefix/", "yolov8n-coco")
    assert "predictions" not in task


def test_build_preannotated_task_requires_dimensions():
    record = make_manifest(image_ids=(9,))[0]
    record.width = None
    with pytest.raises(ValueError):
        build_preannotated_task(record, [detection()], "/prefix/", "yolov8n-coco")
