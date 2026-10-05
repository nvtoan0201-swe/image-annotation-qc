# Annotation Quality Checklist

Quality control has two parts: **automated checks** run by scripts, and
**human review** performed by the annotator. Automated findings are reported,
never auto-fixed; a human decides what to correct.

## Automated checks

Run:

```bash
uv run scripts/validate_annotations.py
```

| # | Check | Severity |
| --- | --- | --- |
| 1 | Annotation file is missing or unreadable | error |
| 2 | Image listed in the manifest has no annotations | warning |
| 3 | Annotation references an image that is not in the manifest | error |
| 4 | Unknown class (not one of the four target classes) | error |
| 5 | `x_min >= x_max` or `y_min >= y_max` (zero/negative size) | error |
| 6 | Coordinates outside the image boundaries | error |
| 7 | Negative coordinates | error |
| 8 | Duplicate annotation (same class, IoU >= 0.95) | warning |
| 9 | Extremely tiny box (side < 4 px or area < 16 px) | warning |
| 10 | Suspiciously huge box (area > 80% of the image) | warning |
| 11 | Unusual annotation count for an image (statistical outlier) | warning |

Additional image-level checks:

```bash
uv run scripts/validate_images.py
```

| Check | Severity |
| --- | --- |
| Missing / unreadable / corrupted image file | error |
| Invalid dimensions | error |
| Duplicate file names in the manifest | error |
| Unexpected file format | warning |
| Content or dimensions differ from the manifest | warning |
| Identical duplicate images | warning |

## Human review checklist

### Per image

- [ ] (model-assisted tasks) Every suggested box was reviewed; false
      positives were deleted, missed objects were added, and nothing was
      submitted unreviewed.
- [ ] Every visible target object is labeled (compare against
      `LABELING_GUIDELINES.md` section 1).
- [ ] No forbidden objects are labeled (other classes, statues, depictions,
      reflections).
- [ ] The class is correct (section 6).
- [ ] Each box tightly covers the object: no large background margins, no
      cropped-off parts that are visible.
- [ ] Occluded objects are boxed to the visible silhouette only.
- [ ] Truncated objects stop at the image border.
- [ ] One box per object: nothing split, nothing merged.
- [ ] Similar objects across images are labeled consistently.
- [ ] Unsure cases were skipped and recorded in `ANNOTATION_NOTES.md`.

### Per dataset

- [ ] All images in the manifest have been reviewed.
- [ ] Every image flagged by the automated report has been looked at.
- [ ] A second pass over a random sample (at least 20% of images) happened
      after a break, not immediately after the first pass.
- [ ] Class counts were compared against the reference COCO labels
      (`scripts/reference_agreement.py`) as a *signal only*; differences were
      reviewed, not automatically "fixed".
- [ ] Export was only produced after `validate_annotations.py` reported zero
      errors.

## Review order

1. Fix all **errors** first (invalid geometry, unknown classes, missing files).
2. Review all **warnings**, decide per case: correct, acceptable, or ignore.
   Record decisions in `ANNOTATION_NOTES.md`.
3. Run the human checklist on flagged images, then on the random sample.
4. Re-run the automated checks; they must be error-free before export.
