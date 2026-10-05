# Image Annotation & Quality Control Pipeline

A small, reproducible pipeline that takes a raw image pool and turns it into a
clean, validated bounding-box dataset:

```
raw images -> dataset preparation -> MANUAL annotation -> quality control
           -> validation -> statistics -> clean YOLO export
```

This is **not** a model-training project. It demonstrates the data side of
Computer Vision: dataset preparation, labeling guidelines, annotation
consistency, automated quality control, dataset validation, statistics and
reproducible export.

## Status

| Stage | State |
| --- | --- |
| Dataset download + preparation | done, reproducible |
| Image validation | done, tested |
| Labeling guidelines + Label Studio setup | done |
| **Manual annotation of 75 images** | **MANUAL — to be completed by the annotator** |
| Annotation QC + statistics + export | implemented and unit-tested; real reports pending the manual step |

No annotation results or quality metrics are reported until the manual
annotation has actually been performed. Numbers in this README are only from
commands that were really run.

## Why annotation quality matters

Object detectors can only be as good as the labels they learn from. Boxes that
are loose, missing, duplicated or assigned the wrong class teach the model the
wrong thing. Good datasets are built with explicit rules, consistency checks
and a documented review process — which is what this project implements and
demonstrates end to end.

## Dataset

- **Source:** [coco128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip),
  a 128-image subset of **MS COCO 2017** (`train2017`) distributed by
  Ultralytics as a single ~7 MB archive.
- **License:** images are **CC BY 4.0**; the original COCO annotations are
  CC BY 4.0 (© Microsoft). See `data/README.md`.
- **Raw pool:** 128 images, SHA-256 pinned in `configs/classes.yaml`.
- **Selected for annotation:** **75 images** that contain at least one object
  of the target classes, according to the original COCO reference labels.
- **Classes (4):** `person`, `car`, `dog`, `bicycle`.
- **Why this dataset:** small enough to annotate in about a day, real photos
  with occlusion/truncation/scale variation, clear licensing, stable download.

The reference labels are used only to choose the 75 images (and, optionally,
for the agreement check). **They are not the annotations of this project.**
All annotations in `data/annotations/annotations.json` are created manually in
Label Studio following the guidelines.

## Workflow and commands

### 1. Setup (clean machine)

Requirements: `uv` (manages Python >= 3.11) and Docker (annotation step only).

```bash
git clone <your-repo-url> image-annotation-qc
cd image-annotation-qc
uv sync                       # creates .venv and installs dependencies
uv run pytest                 # 54 tests, no dataset needed
```

### 2. Dataset preparation

```bash
uv run scripts/download_data.py                # ~7 MB, checksum verified
uv run scripts/prepare_dataset.py              # manifest + Label Studio tasks
uv run scripts/validate_images.py              # file/dimension/duplicate checks
```

Add `--with-reference` to the download to also extract the original COCO
labels for the optional agreement check.

### 3. **MANUAL** annotation

This step cannot be automated and is not pretended to be complete.

1. Start Label Studio (Docker):

   ```bash
   ./scripts/label_studio.sh start          # http://localhost:8080
   # or, with the compose plugin:
   # docker compose -f annotation/docker-compose.yml up -d
   ```

   Verified on this machine: Label Studio 1.23.2 starts, serves `/health`
   (HTTP 200) and serves local files only to authenticated sessions
   (Docker 29, Ubuntu 26.04). `./scripts/label_studio.sh reset` removes the
   container and all local Label Studio state.

2. Open **http://localhost:8080** and create a local account on first run
   (`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` is already configured;
   all data stays in `.labelstudio/`, which is git-ignored).
3. Create a project named **`coco128-subset-bbox`**. In
   *Settings → Labeling Interface → Code*, paste the content of
   `annotation/label_studio_config.xml`. Classes: `person`, `car`, `dog`,
   `bicycle` (bounding boxes).
4. Import `data/annotations/tasks.json` (75 tasks referencing the local
   images in `data/raw/images/`).
5. Annotate **all 75 images** according to
   `annotation/LABELING_GUIDELINES.md`. Record unsure cases in
   `annotation/ANNOTATION_NOTES.md`.
6. Export the project as **COCO**: *Project → Export → COCO → Export*. Put the
   JSON from the downloaded archive at **`data/annotations/annotations.json`**.
7. Review your own work with `annotation/QUALITY_CHECKLIST.md` (human checks)
   and then run the automated QC below.

### 4. Quality control

```bash
uv run scripts/validate_annotations.py     # errors/warnings report, nothing auto-fixed
uv run scripts/analyze_dataset.py          # statistics + two plots
uv run scripts/reference_agreement.py      # optional signal vs COCO labels
```

`validate_annotations.py` checks: unknown classes, `x_min >= x_max`,
`y_min >= y_max`, out-of-bounds and negative coordinates, duplicate boxes
(IoU >= 0.95), tiny boxes, huge boxes, unannotated images and annotation-count
outliers. Findings are reported for a human to decide on — never deleted.

Reports are written to `reports/` (git-ignored):
`annotation_qc.json` + `annotation_qc.md`, `dataset_report.json`,
`figures/annotations_per_image.png`, `figures/class_distribution.png`,
`reference_agreement.json`.

### 5. Export

```bash
uv run scripts/export_dataset.py
```

Validation runs first; the export **refuses to run when any error is found**.
On success it writes a YOLO dataset to `data/processed/yolo/`:

```
data/processed/yolo/
├── images/              # copies of the 75 raw images
├── labels/              # one .txt per image: class_id cx cy w h (normalized)
├── classes.txt          # person, car, dog, bicycle
├── dataset.yaml         # train/val point to images/, names map for YOLO
└── export_summary.json  # real counts from this export
```

## Annotation guidelines (summary)

Full rules: `annotation/LABELING_GUIDELINES.md`.

- Label only `person`, `car`, `dog`, `bicycle`; label every qualifying
  instance; label partially visible and truncated objects down to ~20%
  visibility.
- Do not label depictions (photos/posters/screens), reflections, statues,
  toys, or non-target classes.
- Boxes are tight, axis-aligned, inside the image, one per object; occluded
  objects are boxed to the visible silhouette; truncated objects stop at the
  image border.
- Skip objects with a box side < 4 px; never enlarge a box to make it count.
- When unsure: skip the object and record it in `annotation/ANNOTATION_NOTES.md`.

## Project structure

```
image-annotation-qc/
├── configs/classes.yaml          # dataset, classes, QC thresholds, paths
├── configs/selected_images.txt   # the 75 selected images (selection is documented)
├── data/
│   ├── raw/images/               # downloaded raw pool (git-ignored)
│   ├── reference/labels/         # optional COCO reference labels (git-ignored)
│   ├── processed/images.csv      # dataset manifest (committed)
│   └── annotations/tasks.json    # Label Studio tasks (committed)
├── annotation/                   # guidelines, checklist, Label Studio config + compose
├── scripts/                      # CLI entry points (download -> export)
├── src/annotation_qc/            # library: schemas, io, validators, statistics,
│                                 #          exporter, agreement
├── reports/                      # generated reports (git-ignored)
└── tests/                        # pytest suite
```

## Quality checks at a glance

| Layer | What is checked | Tool |
| --- | --- | --- |
| Images | existence, decoding, dimensions, format, duplicate names/content | `validate_images.py` |
| Annotations | geometry, classes, duplicates, outliers, coverage | `validate_annotations.py` |
| Human | completeness, tightness, consistency, ambiguous cases | `QUALITY_CHECKLIST.md` |
| Export | refuses invalid annotations, writes class mapping | `export_dataset.py` |
| Code | 54 unit tests for validators, statistics, IO, export, agreement | `pytest` |

## Example output (real runs)

Dataset download and preparation:

```
$ uv run scripts/download_data.py --with-reference
Dataset download
  source      : Ultralytics coco128 (subset of MS COCO 2017, train2017 images)
  license     : CC BY 4.0
  downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
  archive     : 6.66 MB (sha256 verified)
  images      : 128 extracted to data/raw/images
  reference   : 128 COCO label files in data/reference/labels
  raw pool    : 128 images ready

$ uv run scripts/prepare_dataset.py
Dataset preparation
  raw pool     : 128 images in data/raw/images
  selected     : 75 images
  status       : 75 ok, 0 missing, 0 unreadable, 0 duplicate
  classes      : person, car, dog, bicycle
  manifest     : data/processed/images.csv
  tasks        : data/annotations/tasks.json (75 tasks)

$ uv run scripts/validate_images.py
Image validation
  checked      : 75 images
  errors       : 0
  warnings     : 0
  report       : reports/image_validation.json
  result       : PASS
```

Tests:

```
$ uv run pytest
54 passed in 0.11s
```

Annotation-dependent outputs (`validate_annotations.py`, `analyze_dataset.py`,
`export_dataset.py`) are intentionally not shown until the manual annotation
is complete.

## Lessons learned

Real issues encountered while building the pipeline:

1. **Small public datasets can be internally inconsistent.** coco128 ships
   128 images and 128 label files, but two labels have no image
   (`000000000656`, `000000000659`) and two images have no label
   (`000000000250`, `000000000508`). A naive selection by label file produced
   broken tasks; requiring the image to exist fixed it. This is exactly why
   the manifest and validators exist instead of "just downloading images".
2. **Pin and verify downloads.** The archive SHA-256 is stored in the config
   and checked on every download, so silent upstream changes cannot corrupt
   the dataset.
3. **Local annotation servers need care.** Label Studio serves local files only
   with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and a document root;
   the container runs as UID 1001, so the state directory needs explicit
   permissions. The Docker Compose plugin is not installed on every machine,
   so `scripts/label_studio.sh` provides an equivalent `docker run` workflow.
4. **Reference annotations are a signal, not ground truth.** The COCO labels
   used for image selection are heavily imbalanced (person dominates) and may
   miss small objects. The agreement check therefore reports differences
   without assuming the reference is right.
5. **Coordinate-format bugs are easy to introduce.** COCO `[x, y, w, h]` and
   YOLO normalized centers are converted in exactly one place (`schemas.py`,
   `exporter.py`), and validators catch the classic symptoms
   (`x_min >= x_max`, out-of-bounds boxes).

Lessons from the manual annotation itself (consistency problems found,
ambiguous cases decided, time spent per class) will be added here after the
MANUAL step is completed honestly.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (see `data/README.md`).
