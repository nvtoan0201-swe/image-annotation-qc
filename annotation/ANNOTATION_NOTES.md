# Annotation Notes

MANUAL FILE — filled in by the annotator during and after labeling.

Record every unsure case here instead of guessing. Automated validators also
never fix anything silently; if you decide to keep a flagged annotation, note
the decision here with a one-line reason.

## Unsure / skipped objects

| Date | Image | What I saw | Decision |
| --- | --- | --- | --- |
| | | | |

## Decisions on automated warnings

| Date | Image / annotation | Warning | Decision |
| --- | --- | --- | --- |
| | | | |

## Session log

| Date | What happened | Decision |
| --- | --- | --- |
| 2026-10-05 | First fully-manual pass produced 60 labeled images / 153 boxes (person 128, car 15, dog 8, bicycle 2). 15 tasks had been submitted empty. | Backed up to `data/annotations/manual_annotations_backup.json` (not part of the final dataset; git-ignored), then reset the project for one uniform pass. |
| 2026-10-05 | YOLOv8n pre-annotation run: 220 detections in 70 of 75 images (person 198, dog 11, car 9, bicycle 2); 5 images with no detections. | Apply model pre-annotations to all 75 tasks; every box must be reviewed and corrected by hand, and the 5 empty images labeled from scratch. |

## General observations

- The pretrained model misses small and occluded objects, so pre-annotations
  are only a starting point. The final annotations are the reviewed result,
  never the raw predictions.
- (example format) Some persons are only visible through windows; skipped as
  depictions per guidelines section 2.
