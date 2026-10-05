"""Automated checks for images and annotations.

Validators only report findings. They never modify or delete data: every
finding is meant to be reviewed by a human, who decides what to fix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from annotation_qc.io import inspect_image
from annotation_qc.schemas import ImageRecord

ERROR = "error"
WARNING = "warning"

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Finding:
    """One issue (or informational note) produced by a validator."""

    check: str
    severity: str
    message: str
    image_id: int | None = None
    annotation_id: int | None = None
    filename: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    counts = {ERROR: 0, WARNING: 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def validate_image_files(records: list[ImageRecord], image_dir: Path) -> list[Finding]:
    """Validate that every manifest entry maps to a readable, unique image."""
    findings: list[Finding] = []

    by_name: dict[str, list[ImageRecord]] = defaultdict(list)
    by_checksum: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_name[record.filename].append(record)

    for filename, group in sorted(by_name.items()):
        if len(group) > 1:
            ids = ", ".join(str(record.image_id) for record in group)
            findings.append(
                Finding(
                    check="duplicate_filename",
                    severity=ERROR,
                    message=f"file name appears {len(group)} times in the manifest (image ids: {ids})",
                    filename=filename,
                )
            )

    for record in sorted(records, key=lambda item: item.image_id):
        path = image_dir / record.filename
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            findings.append(
                Finding(
                    check="unexpected_format",
                    severity=WARNING,
                    message=f"unexpected file extension '{suffix}'",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        info = inspect_image(path)
        if info.error:
            findings.append(
                Finding(
                    check="image_unreadable",
                    severity=ERROR,
                    message=info.error,
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
            continue
        if not info.width or not info.height or info.width <= 0 or info.height <= 0:
            findings.append(
                Finding(
                    check="invalid_dimensions",
                    severity=ERROR,
                    message=f"invalid image dimensions {info.width}x{info.height}",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
            continue
        if record.width and record.height and (info.width, info.height) != (record.width, record.height):
            findings.append(
                Finding(
                    check="dimensions_changed",
                    severity=WARNING,
                    message=(
                        f"manifest says {record.width}x{record.height}, file is "
                        f"{info.width}x{info.height}; re-run prepare_dataset.py"
                    ),
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        if record.sha256 and info.sha256 != record.sha256:
            findings.append(
                Finding(
                    check="content_changed",
                    severity=WARNING,
                    message="file content differs from the manifest checksum",
                    image_id=record.image_id,
                    filename=record.filename,
                )
            )
        by_checksum[info.sha256].append(record)

    for checksum, group in sorted(by_checksum.items()):
        if len(group) > 1:
            names = ", ".join(record.filename for record in group)
            findings.append(
                Finding(
                    check="duplicate_content",
                    severity=WARNING,
                    message=f"{len(group)} images share identical content: {names}",
                    image_id=group[0].image_id,
                )
            )

    return findings
