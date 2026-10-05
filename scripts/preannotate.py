#!/usr/bin/env python3
"""Generate model-assisted pre-annotations with a pretrained detector.

The optional `preannotate` dependency group provides Ultralytics YOLO. This
script runs a COCO-pretrained model on the selected images and writes
`data/annotations/tasks_preannotated.json`, which is the same task file as
`tasks.json` plus a Label Studio `predictions` entry per image.

Pre-annotations are SUGGESTIONS. Import them into Label Studio, then review,
correct and complete every image by hand. Never submit them unreviewed, and
never claim unreviewed model output as manual annotation.

Commands:
    uv sync --group preannotate
    uv run --group preannotate scripts/preannotate.py
    uv run --group preannotate scripts/preannotate.py --limit 5 --conf 0.3
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from annotation_qc.io import (  # noqa: E402
    find_project_root,
    load_config,
    local_file_url_prefix,
    read_manifest,
    write_json,
)
from annotation_qc.preannotate import Detection, build_preannotated_task  # noqa: E402

DEFAULT_MODEL = "models/yolov8n.pt"


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: optional pre-annotation dependencies are not installed.")
        print("       run: uv sync --group preannotate")
        raise SystemExit(2)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(model_path))


def detect(model, image_path, target_names: set[str], conf: float, iou: float) -> list[Detection]:
    result = model.predict(str(image_path), conf=conf, iou=iou, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        if class_name not in target_names:
            continue
        x_min, y_min, x_max, y_max = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                class_name=class_name,
                confidence=float(box.conf),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path (downloaded if missing)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images (testing)")
    parser.add_argument("--output", default=None, help="output task file (default from config)")
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]
    target_names = {category["name"] for category in config["classes"]}
    records = [
        record
        for record in read_manifest(root / paths["manifest"])
        if record.is_ok and record.width and record.height
    ]
    if args.limit:
        records = records[: args.limit]

    print("Model-assisted pre-annotation")
    print(f"  model           : {args.model} (conf={args.conf}, iou={args.iou})")
    print(f"  images          : {len(records)}")

    model = load_model(root / args.model)
    model_version = f"{os.path.basename(args.model).rsplit('.', 1)[0]}-coco"
    url_prefix = local_file_url_prefix(config)

    per_class: Counter[str] = Counter()
    confidences: list[float] = []
    images_without_detections = 0
    tasks = []
    for record in records:
        detections = detect(
            model, root / paths["raw_images"] / record.filename, target_names, args.conf, args.iou
        )
        if not detections:
            images_without_detections += 1
        for detection in detections:
            per_class[detection.class_name] += 1
            confidences.append(detection.confidence)
        tasks.append(build_preannotated_task(record, detections, url_prefix, model_version))

    output_path = root / (args.output or paths["preannotated_tasks"])
    write_json(output_path, tasks)

    total = sum(per_class.values())
    print(f"  detections      : {total} in {len(records) - images_without_detections} images")
    if per_class:
        print("  per class       : " + ", ".join(
            f"{name}={count}" for name, count in per_class.most_common()
        ))
    if confidences:
        print(f"  mean confidence : {sum(confidences) / len(confidences):.3f}")
    print(f"  no detections   : {images_without_detections} images (annotate these from scratch)")
    print(f"  output          : {output_path.relative_to(root)}")
    print("  reminder        : suggestions only - review, correct and complete every box by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
