# Labeling Guidelines

Project: coco128 subset (75 images) · Task: bounding boxes · Version 1.0

Target classes: **person, car, dog, bicycle**.

The goal of these guidelines is consistency. When two situations are similar,
they must be annotated the same way. If these rules do not cover a case, skip
the object and record it in `annotation/ANNOTATION_NOTES.md` instead of
guessing.

## 0. If the image has model pre-annotations

Some tasks may arrive with suggested boxes from a pretrained model
(`tasks_preannotated.json`). They are **suggestions, not annotations**:

- Verify every suggested box: is the class correct, is the box tight, is it a
  real object (not a depiction or reflection) per sections 2 and 6?
- Delete false positives and duplicates.
- Add every target object the model missed, especially small or occluded ones.
- Inspect low-confidence suggestions with extra care.
- Submitting predictions without reviewing them is a QC failure.

All other rules are unchanged: the human is responsible for every box in the
final dataset.

## 1. What to label

Label every instance of a target class that is visible in the image and meets
the rules below, including:

- fully visible objects,
- partially visible / occluded objects (at least ~20% visible and the class
  must be clear),
- objects cut off by the image border (truncated).

Review every image. An image without any target object is allowed, but it must
be an honest "no target present" decision.

## 2. What NOT to label

- Any class that is not `person`, `car`, `dog` or `bicycle` (for example:
  motorcycle, truck, bus, cat, horse).
- Statues, mannequins, dolls, toys, cartoon characters (not `person`).
- Objects that exist only as a depiction: photos, posters, paintings, TV or
  monitor images, T-shirt prints, stickers.
- Objects visible only as a reflection (for example a car in a shop window).
- Unidentifiable blobs, silhouettes or parts that cannot be assigned a class
  with confidence.
- Objects below the minimum size rule (section 5).

## 3. Drawing boxes

- Axis-aligned rectangle, **tight around the visible extent** of the object.
- Include the whole object: for a person head/hair, body, arms and legs; for a
  car wheels and mirrors; for a dog head, legs and tail; for a bicycle both
  wheels and handlebar.
- One box per object. Do not split an object into several boxes (person +
  head). Do not merge several objects into one box (a group of people).
- Boxes must stay inside the image: `x_min >= 0`, `y_min >= 0`,
  `x_max <= width`, `y_max <= height`.
- Avoid large background margins; up to ~3 px margin is acceptable.
- Do not box shadows separately; do not box a bag carried by a person as a
  separate object unless the bag itself is a target class.

## 4. Occlusion and truncation

- **Occluded object:** box the visible silhouette. Do not extrapolate the
  hidden part. If less than ~20% of the object is visible, or the class cannot
  be confirmed, skip it.
- **Truncated object (image border):** box the visible part and stop at the
  image edge. Never draw outside the image.
- **Occluded and truncated:** same rule, use the visible extent.

## 5. Small objects

- Skip objects whose box would have a side shorter than **4 px**. Automated QC
  flags every box with a side < 4 px or area < 16 px.
- A small but clearly identifiable object may be labeled if a tight box with
  side >= 4 px is possible.
- Never enlarge a box to make it "count": boxes must stay tight.

## 6. Class definitions

| Class | Include | Do not include |
| --- | --- | --- |
| `person` | Real humans of any age or clothing, seated, standing, occluded or truncated | Statues, dolls, mannequins, cartoons, depictions |
| `car` | Passenger cars, SUVs, vans, taxis, police cars | Trucks, buses, motorcycles, bicycles, toy cars |
| `dog` | Domestic dogs (any breed, any pose) | Cats, wolves, foxes, other animals, toy dogs |
| `bicycle` | Pedal bicycles (including e-bikes with pedals) | Motorcycles, scooters, mopeds, bicycle depictions |

## 7. Ambiguous cases

- **bicycle vs. motorcycle:** pedals and no engine → `bicycle`; engine and no
  pedals → skip. Cannot tell → skip.
- **car vs. truck/van:** vans are `car`; box trucks, buses and heavy vehicles
  are skipped (not target classes). If the shape is unclear, prefer `car` only
  when windows and passenger-car proportions are visible, otherwise skip.
- **dog vs. other animal:** only domestic dogs. Unsure → skip.
- **person vs. depiction:** if it is printed, painted or displayed on a screen
  → skip.

## 8. When unsure

1. Skip the object.
2. Write one line in `annotation/ANNOTATION_NOTES.md`: image file name, what
   you saw, why you were unsure.
3. Continue. Do not guess and do not invent classes.

If a whole image is unusable (extremely blurry, wrong content), submit it with
no boxes and record it in the notes file.

## 9. Suggested workflow

1. Annotate in task order; keep the right sidebar zoom on.
2. After every ~20 images, re-read sections 2 and 6 to stay consistent.
3. Do not correct old images while tired; finish the batch, then run
   `scripts/validate_annotations.py` and use the human checklist for review.
