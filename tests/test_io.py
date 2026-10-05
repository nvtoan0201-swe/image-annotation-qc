"""Tests for configuration, manifest, task and COCO IO helpers."""

from __future__ import annotations

import json
import zipfile

import pytest

from annotation_qc.io import (
    CocoFormatError,
    find_project_root,
    inspect_image,
    load_coco,
    load_config,
    read_manifest,
    read_selected_images,
    write_label_studio_tasks,
    write_manifest,
)
from annotation_qc.schemas import AnnotationFile, ImageRecord


def make_coco_payload() -> dict:
    return {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 80}],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bicycle"},
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]}],
    }


def test_manifest_roundtrip(tmp_path):
    records = [
        ImageRecord(
            image_id=7,
            filename="000000000007.jpg",
            source="coco128-subset",
            license="CC BY 4.0",
            width=640,
            height=480,
            sha256="abc123",
            status="ok",
            note="",
        )
    ]
    path = tmp_path / "images.csv"
    write_manifest(path, records)
    loaded = read_manifest(path)
    assert loaded == records


def test_manifest_roundtrip_with_missing_dimensions(tmp_path):
    records = [ImageRecord(image_id=9, filename="000000000009.jpg", status="missing", note="file not found")]
    path = tmp_path / "images.csv"
    write_manifest(path, records)
    loaded = read_manifest(path)
    assert loaded[0].width is None
    assert loaded[0].status == "missing"


def test_write_label_studio_tasks(tmp_path):
    records = [
        ImageRecord(image_id=1, filename="000000000001.jpg", status="ok"),
        ImageRecord(image_id=2, filename="000000000002.jpg", status="ok"),
    ]
    path = tmp_path / "tasks.json"
    count = write_label_studio_tasks(path, records, "/data/local-files/?d=raw/")
    tasks = json.loads(path.read_text())
    assert count == 2
    assert tasks[0]["id"] == 1
    assert tasks[0]["data"]["image"] == "/data/local-files/?d=raw/000000000001.jpg"


def test_read_selected_images_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "selected.txt"
    path.write_text("# comment\n\n0001.jpg\n 0002.jpg \n")
    assert read_selected_images(path) == ["0001.jpg", "0002.jpg"]


def test_inspect_image_valid(tmp_path):
    from helpers import write_image

    path = tmp_path / "image.png"
    write_image(path, size=(32, 24))
    info = inspect_image(path)
    assert info.is_valid
    assert (info.width, info.height) == (32, 24)
    assert info.image_format == "PNG"
    assert len(info.sha256) == 64


def test_inspect_image_missing(tmp_path):
    info = inspect_image(tmp_path / "nope.jpg")
    assert not info.is_valid
    assert info.error == "file not found"


def test_inspect_image_corrupted(tmp_path):
    path = tmp_path / "broken.png"
    path.write_text("not an image")
    info = inspect_image(path)
    assert not info.is_valid


def test_load_coco_from_json(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(make_coco_payload()))
    annotation_file = load_coco(path)
    assert isinstance(annotation_file, AnnotationFile)
    assert annotation_file.images[1].width == 100
    assert annotation_file.annotations[0].x_min == 10
    assert annotation_file.annotations[0].x_max == 40
    assert annotation_file.category_name(0) == "person"


def test_load_coco_from_zip(tmp_path):
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.json", json.dumps(make_coco_payload()))
    annotation_file = load_coco(path)
    assert len(annotation_file.annotations) == 1


def test_load_coco_from_zip_with_multiple_json_raises(tmp_path):
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.json", json.dumps(make_coco_payload()))
        archive.writestr("extra.json", "{}")
    with pytest.raises(CocoFormatError):
        load_coco(path)


def test_load_coco_missing_keys_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"images": []}))
    with pytest.raises(CocoFormatError):
        load_coco(path)


def test_load_coco_invalid_bbox_raises(tmp_path):
    payload = make_coco_payload()
    payload["annotations"][0]["bbox"] = [1, 2, 3]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(CocoFormatError):
        load_coco(path)


def test_project_config_is_consistent():
    root = find_project_root()
    assert root.name == "image-annotation-qc"
    config = load_config(root / "configs" / "classes.yaml")
    assert [category["id"] for category in config["classes"]] == [0, 1, 2, 3]
    assert [category["name"] for category in config["classes"]] == ["person", "car", "dog", "bicycle"]
    for key in ("min_box_side_px", "min_box_area_px", "max_box_area_fraction", "duplicate_iou_threshold"):
        assert key in config["quality_control"]
