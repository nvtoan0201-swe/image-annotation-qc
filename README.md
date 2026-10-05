# Image Annotation & Quality Control Pipeline

Turns a raw image pool into a clean, validated bounding-box dataset. This is
**not** a model-training project: it demonstrates the data side of Computer
Vision — dataset preparation, labeling guidelines, annotation, quality
control, validation, statistics and export.

```
raw images -> dataset preparation -> annotation in Label Studio
           -> (optional) pretrained-model pre-annotations, reviewed by a human
           -> automated QC -> statistics -> validated YOLO export
```

**Status:** the pipeline is implemented and tested (63 tests). Annotation
supports two modes: fully manual, or model-assisted (a pretrained YOLO model
proposes boxes that a human must review and correct). No annotation metrics
are reported until the human review and validation are complete.

## Dataset

- [coco128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip):
  128 images from MS COCO 2017, one ~7 MB archive, SHA-256 pinned in
  `configs/classes.yaml`. Raw images are downloaded, not committed.
- License: images **CC BY 4.0** (MS COCO).
- 75 images containing `person`, `car`, `dog` or `bicycle` are selected;
  the deterministic selection is stored in `configs/selected_images.txt`.
- The original COCO labels were used **only** to choose those images and,
  optionally, for an agreement check. They are never used as this project's
  annotations.

## Requirements

`uv` (installs Python >= 3.11 automatically) and Docker for the annotation
step only.

## How to run

### 1. Setup

```bash
uv sync
uv run pytest
```

### 2. Prepare the dataset

```bash
uv run scripts/download_data.py      # add --with-reference for the agreement check
uv run scripts/prepare_dataset.py    # manifest + 75 Label Studio tasks
uv run scripts/validate_images.py
```

### 2b. Optional: model-assisted pre-annotations

```bash
uv sync --group preannotate
uv run --group preannotate scripts/preannotate.py
```

This runs a COCO-pretrained YOLOv8n model (CPU) over the selected images and
writes `data/annotations/tasks_preannotated.json`: the same tasks plus Label
Studio `predictions`. Real numbers from this run: 220 detections in 70 of 75
images (person 198, dog 11, car 9, bicycle 2), mean confidence 0.64, and 5
images where the model found nothing.

**Honesty rule:** pre-annotations are model suggestions, not labels. If you use
them, the correct claim is *"reviewed and corrected model pre-annotations"* —
never *"fully manual annotation"*. Human review is mandatory, and a task
submitted without reviewing its predictions is a QC failure.

### 3. Annotate

1. `./scripts/label_studio.sh start` → http://localhost:8080 (create a local
   account on first run). With the compose plugin:
   `docker compose -f annotation/docker-compose.yml up -d`.
2. Create the project `coco128-subset-bbox` and paste
   `annotation/label_studio_config.xml` into *Settings → Labeling Interface →
   Code* (classes: `person`, `car`, `dog`, `bicycle`).
3. Add the local files connection so Label Studio may serve the mounted
   images: *Settings → Cloud Storage → Add Source Storage → Local Files*, set
   the path to `/label-studio/data/data`, click **Save** (do not sync — the
   tasks come from `tasks.json`). Since Label Studio 1.23 the
   `/data/local-files/` endpoint only serves files that belong to a
   configured storage connection.
4. Import `data/annotations/tasks.json` (fully manual) or
   `data/annotations/tasks_preannotated.json` (model-assisted).
5. Annotate all **75 images** following `annotation/LABELING_GUIDELINES.md`;
   for pre-annotated tasks, verify every suggested box, delete false
   positives, add missed objects, and only then submit. Record unsure cases
   in `annotation/ANNOTATION_NOTES.md`.
6. Export COCO (*Project → Export → COCO*) and save the JSON as
   `data/annotations/annotations.json`.
7. `./scripts/label_studio.sh stop`.

### 4. Quality control

```bash
uv run scripts/validate_annotations.py    # report only, never auto-fixes
uv run scripts/analyze_dataset.py         # statistics + two plots
uv run scripts/reference_agreement.py     # optional signal, not ground truth
```

### 5. Export

```bash
uv run scripts/export_dataset.py          # refuses to export on validation errors
```

Output: `data/processed/yolo/` with `images/`, `labels/` (`class_id cx cy w h`,
normalized), `classes.txt`, `dataset.yaml` and `export_summary.json`.

## What is checked

- **Images:** existence, decoding, dimensions, format, duplicate names/content.
- **Annotations:** unknown class, `x_min >= x_max`, `y_min >= y_max`,
  out-of-bounds and negative coordinates, duplicate boxes (IoU >= 0.95), tiny
  and huge boxes, images without annotations, annotation-count outliers.
- **Human review:** `annotation/QUALITY_CHECKLIST.md`.

Findings are always reported for a human decision — nothing is deleted or
"fixed" automatically. Reports are written to `reports/` (git-ignored).

## Project structure

```
configs/            classes, QC thresholds, selected image list
data/               raw pool (ignored), manifest, tasks, manual annotations
annotation/         guidelines, checklist, notes, Label Studio config/compose
models/             pretrained weights (downloaded on demand, ignored)
scripts/            CLI entry points
src/annotation_qc/  schemas, io, validators, statistics, exporter, agreement,
                    preannotate
tests/              63 pytest tests
reports/            generated reports and plots (ignored)
```

## Verified so far (real output)

```
$ uv run scripts/prepare_dataset.py
  selected     : 75 images
  status       : 75 ok, 0 missing, 0 unreadable, 0 duplicate
  tasks        : data/annotations/tasks.json (75 tasks)

$ uv run --group preannotate scripts/preannotate.py
Model-assisted pre-annotation
  model           : models/yolov8n.pt (conf=0.25, iou=0.7)
  images          : 75
  detections      : 220 in 70 images
  per class       : person=198, dog=11, car=9, bicycle=2
  mean confidence : 0.638
  no detections   : 5 images (annotate these from scratch)
  output          : data/annotations/tasks_preannotated.json

$ uv run pytest
63 passed
```

## Notes

- coco128 is internally inconsistent: two label files have no image and two
  images have no label. The selection rule requires the image to exist, which
  is why 75 tasks remain.
- The COCO reference labels are a signal, not ground truth: the agreement
  check reports differences for human review.
- Pre-annotations change the work, not the risk: the model found no target at
  all in 5 images and misses small or occluded objects, so every pre-annotated
  task still requires a full human pass.

## License

Code: MIT (`LICENSE`). Images: CC BY 4.0 (MS COCO).
