#!/usr/bin/env python3
"""Validate the raw image files listed in the dataset manifest.

Checks file existence, readability, dimensions, file format, duplicate file
names and duplicate content. Findings are reported, never fixed automatically.
"""

from __future__ import annotations

import argparse

from annotation_qc.io import find_project_root, load_config, read_manifest, write_json
from annotation_qc.validators import ERROR, WARNING, count_by_severity, validate_image_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    root = find_project_root()
    config = load_config(root / "configs" / "classes.yaml")
    paths = config["paths"]

    records = read_manifest(root / paths["manifest"])
    findings = validate_image_files(records, root / paths["raw_images"])
    counts = count_by_severity(findings)

    report = {
        "manifest_images": len(records),
        "errors": counts[ERROR],
        "warnings": counts[WARNING],
        "findings": [finding.to_dict() for finding in findings],
    }
    report_path = root / paths["reports"] / "image_validation.json"
    write_json(report_path, report)

    print("Image validation")
    print(f"  checked      : {len(records)} images")
    print(f"  errors       : {counts[ERROR]}")
    print(f"  warnings     : {counts[WARNING]}")
    print(f"  report       : {report_path.relative_to(root)}")
    for finding in findings:
        prefix = "ERROR  " if finding.severity == ERROR else "WARNING"
        location = finding.filename or (f"image {finding.image_id}" if finding.image_id else "")
        print(f"    {prefix} [{finding.check}] {location}: {finding.message}")
    print(f"  result       : {'FAIL' if counts[ERROR] else 'PASS'}")
    return 1 if counts[ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
