#!/usr/bin/env python3
"""Compute dataset statistics after annotation.

Writes reports/dataset_report.json and, unless --no-plots is given, two simple
figures under reports/figures/. All numbers come from the real manifest and
the real annotation file.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from annotation_qc.io import (  # noqa: E402
    CocoFormatError,
    find_project_root,
    load_coco,
    load_config,
    read_manifest,
    write_json,
)
from annotation_qc.statistics import build_dataset_report, counts_by_image  # noqa: E402
from annotation_qc.validators import (  # noqa: E402
    ERROR,
    WARNING,
    count_by_severity,
    validate_annotations,
)


def save_plots(per_image_counts: list[int], per_class: dict[str, int], figures_dir) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    figure_path = figures_dir / "annotations_per_image.png"
    figure, axis = plt.subplots(figsize=(7, 4))
    bins = range(0, max(per_image_counts, default=0) + 2)
    axis.hist(per_image_counts, bins=bins, edgecolor="white")
    axis.set_xlabel("annotations per image")
    axis.set_ylabel("images")
    axis.set_title("Annotations per image")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    paths.append(figure_path)

    figure_path = figures_dir / "class_distribution.png"
    figure, axis = plt.subplots(figsize=(7, 4))
    names = list(per_class.keys())
    values = list(per_class.values())
    axis.bar(names, values, color="#4c72b0")
    axis.set_ylabel("annotations")
    axis.set_title("Class distribution")
    for index, value in enumerate(values):
        axis.text(index, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    paths.append(figure_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None, help="override the annotation file path")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]
    annotation_path = root / (args.annotations or paths["annotations"])

    print("Dataset statistics")
    if not annotation_path.is_file():
        print("  MANUAL STEP NOT COMPLETE")
        print(f"  expected annotation file: {paths['annotations']}")
        return 1

    manifest = read_manifest(root / paths["manifest"])
    try:
        annotation_file = load_coco(annotation_path)
    except CocoFormatError as error:
        print(f"  ERROR: {error}")
        return 2

    report = build_dataset_report(annotation_file, manifest)
    findings = validate_annotations(annotation_file, manifest, config)
    counts = count_by_severity(findings)
    report["quality_control"] = {
        "errors": counts[ERROR],
        "warnings": counts[WARNING],
        "report": "reports/annotation_qc.json",
    }
    report_path = root / paths["reports"] / "dataset_report.json"
    write_json(report_path, report)

    images = report["images"]
    per_image = report["annotations_per_image"]
    dimensions = report["image_dimensions"]
    box_area = report["box_size"]["area"] or {}

    print(f"  images          : {images['total']} ({images['annotated']} annotated, "
          f"{images['without_annotations']} without annotations)")
    print(f"  annotations     : {report['annotations']['total']}")
    print("  per class       : " + ", ".join(
        f"{name}={count}" for name, count in report["annotations"]["per_class"].items()
    ))
    print(f"  per image       : min={per_image['min']:.0f}, median={per_image['median']:.0f}, "
          f"p90={per_image['p90']:.0f}, max={per_image['max']:.0f}")
    if dimensions["width"] and dimensions["height"]:
        print(f"  image size      : width median {dimensions['width']['median']:.0f} px, "
              f"height median {dimensions['height']['median']:.0f} px")
    if box_area:
        print(f"  box area px2    : p10={box_area['p10']:.0f}, median={box_area['median']:.0f}, "
              f"p90={box_area['p90']:.0f}")
    print(f"  imbalance ratio : {report['annotations']['class_imbalance_ratio']}")
    print(f"  QC status       : {counts[ERROR]} errors, {counts[WARNING]} warnings "
          "(see reports/annotation_qc.json)")

    if not args.no_plots:
        per_image_counts = list(counts_by_image(
            annotation_file, [record.image_id for record in manifest]
        ).values())
        figures = save_plots(
            per_image_counts, report["annotations"]["per_class"], root / paths["reports"] / "figures"
        )
        print("  figures         : " + ", ".join(
            str(path.relative_to(root)) for path in figures
        ))
    print(f"  report          : {report_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
