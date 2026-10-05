# Data

## Source

The raw image pool is **coco128**, a 128-image subset of the MS COCO 2017
`train2017` images, distributed by Ultralytics as a single small archive:

- Archive: `coco128.zip` (~7 MB)
- URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
- SHA-256: `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- Original dataset: MS COCO 2017, https://cocodataset.org

## License

The images are from MS COCO and are licensed under **Creative Commons
Attribution 4.0 (CC BY 4.0)**. The original COCO annotations are likewise
CC BY 4.0 (© Microsoft). See https://cocodataset.org/#termsofuse.

## Why this dataset

- Small enough to download and fully annotate in about one day.
- Real photographs with common object classes and natural occlusion,
  truncation and object-scale variation.
- Clear licensing and a stable, checksummed download.
- The images already contain objects from several COCO classes, which makes it
  easy to demonstrate "label only the target classes" rules.

## Raw pool vs. selected subset

`download_data.py` downloads all **128** images into `data/raw/images/`.
`configs/selected_images.txt` then selects the **75** images that contain at
least one object of the four target classes `person`, `car`, `dog`, `bicycle`.
The selection was derived from the original COCO labels that ship with
coco128 and is deterministic.

**Archive inconsistency found during preparation:** coco128 ships 128 images
and 128 label files, but two label files (`000000000656`, `000000000659`) have
no matching image and two images (`000000000250`, `000000000508`) have no
matching label. The selection rule therefore requires an image file to exist
as well, which is why 75 images remain instead of 77. This is a real-world
dataset quality issue that the manifest-based preparation catches instead of
silently producing broken tasks.

**Honesty note:** the COCO reference labels were used *only* to choose which
images to include. They are **not** the annotations of this project. Every
bounding box in `data/annotations/annotations.json` is drawn manually in Label
Studio following `annotation/LABELING_GUIDELINES.md`. The reference labels are
not part of the dataset export and are only used for the optional, clearly
labelled agreement check (`scripts/reference_agreement.py`).

## Files

| Path | Description | In git |
| --- | --- | --- |
| `data/raw/images/` | Raw image pool (128 downloaded images) | no |
| `data/reference/labels/` | Optional COCO reference labels (YOLO format) | no |
| `data/processed/images.csv` | Dataset manifest for the 75 selected images | yes |
| `data/annotations/tasks.json` | Label Studio tasks (local file references) | yes |
| `data/annotations/annotations.json` | Manual annotations (COCO format, produced by the annotator) | yes |
| `data/processed/yolo/` | Final validated export | no |

## Manifest fields (`data/processed/images.csv`)

| Field | Meaning |
| --- | --- |
| `image_id` | Stable id, the numeric part of the COCO file name |
| `filename` | File name inside `data/raw/images/` |
| `width`, `height` | Image dimensions in pixels |
| `source` | Dataset name |
| `license` | Image license |
| `sha256` | Content checksum, used for duplicate detection |
| `status` | `ok`, `missing`, `unreadable` or `duplicate` |
| `note` | Details for non-`ok` entries |

## Reproduce

```bash
uv run scripts/download_data.py --with-reference
uv run scripts/prepare_dataset.py
```
