"""Tests for the validated YOLO export."""

from __future__ import annotations

import json

import pytest
import yaml

from helpers import annotation_file, make_config, make_manifest, write_image

from annotation_qc.exporter import ExportRefused, export_yolo_dataset, format_yolo_line
from annotation_qc.schemas import Annotation


def test_format_yolo_line_normalizes_coordinates():
    annotation = Annotation(annotation_id=1, image_id=1, category_id=0, x_min=10, y_min=20, x_max=40, y_max=60)
    assert format_yolo_line(0, annotation, 100, 80) == "0 0.250000 0.500000 0.300000 0.500000"


def test_export_writes_yolo_structure(tmp_path):
    manifest = make_manifest(image_ids=(1, 2))
    for record in manifest:
        write_image(tmp_path / "raw" / record.filename)
    file = annotation_file(
        image_ids=(1, 2),
        boxes=[
            (1, 1, 0, (10, 20, 30, 40)),
            (2, 2, 1, (0, 0, 50, 40)),
        ],
    )
    output = tmp_path / "yolo"
    result = export_yolo_dataset(file, manifest, make_config(), tmp_path / "raw", output)

    assert result.images == 2
    assert result.labels == 2
    assert result.errors == 0
    assert (output / "labels" / "000000000001.txt").read_text() == "0 0.250000 0.500000 0.300000 0.500000\n"
    assert (output / "labels" / "000000000002.txt").read_text() == "1 0.250000 0.250000 0.500000 0.500000\n"
    assert (output / "classes.txt").read_text() == "person\ncar\ndog\nbicycle\n"
    names = yaml.safe_load((output / "dataset.yaml").read_text())["names"]
    assert names == {0: "person", 1: "car", 2: "dog", 3: "bicycle"}
    assert (output / "images" / "000000000001.jpg").is_file()
    summary = json.loads((output / "export_summary.json").read_text())
    assert summary["labels"] == 2
    assert summary["classes"] == ["person", "car", "dog", "bicycle"]


def test_export_writes_empty_label_file_for_unannotated_image(tmp_path):
    manifest = make_manifest(image_ids=(1, 2))
    for record in manifest:
        write_image(tmp_path / "raw" / record.filename)
    file = annotation_file(image_ids=(1, 2), boxes=[(1, 1, 0, (10, 20, 30, 40))])
    output = tmp_path / "yolo"
    result = export_yolo_dataset(file, manifest, make_config(), tmp_path / "raw", output)
    assert result.labels == 1
    assert (output / "labels" / "000000000002.txt").read_text() == ""


def test_export_refuses_invalid_box(tmp_path):
    manifest = make_manifest(image_ids=(1,))
    write_image(tmp_path / "raw" / manifest[0].filename)
    file = annotation_file(image_ids=(1,), boxes=[(1, 1, 0, (10, 10, 0, 20))])
    with pytest.raises(ExportRefused) as error:
        export_yolo_dataset(file, manifest, make_config(), tmp_path / "raw", tmp_path / "yolo")
    assert any(finding.check == "invalid_box_size" for finding in error.value.findings)


def test_export_refuses_unknown_class(tmp_path):
    manifest = make_manifest(image_ids=(1,))
    write_image(tmp_path / "raw" / manifest[0].filename)
    file = annotation_file(
        image_ids=(1,),
        categories={0: "person", 1: "car", 2: "dog", 3: "bicycle", 9: "cat"},
        boxes=[(1, 1, 9, (10, 10, 20, 20))],
    )
    with pytest.raises(ExportRefused):
        export_yolo_dataset(file, manifest, make_config(), tmp_path / "raw", tmp_path / "yolo")


def test_export_removes_stale_files(tmp_path):
    manifest = make_manifest(image_ids=(1,))
    write_image(tmp_path / "raw" / manifest[0].filename)
    output = tmp_path / "yolo"
    output.mkdir()
    (output / "stale.txt").write_text("old")
    file = annotation_file(image_ids=(1,), boxes=[(1, 1, 0, (10, 10, 20, 20))])
    export_yolo_dataset(file, manifest, make_config(), tmp_path / "raw", output)
    assert not (output / "stale.txt").exists()
