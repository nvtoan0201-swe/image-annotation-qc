"""File input/output helpers: config, manifest, images and COCO annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml
from PIL import Image, UnidentifiedImageError

from annotation_qc.schemas import Annotation, AnnotationFile, Category, ImageRecord

MANIFEST_FIELDS = [
    "image_id",
    "filename",
    "width",
    "height",
    "source",
    "license",
    "sha256",
    "status",
    "note",
]


class CocoFormatError(ValueError):
    """Raised when an annotation file is not valid COCO JSON."""


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root (the directory containing pyproject.toml)."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("could not locate the project root (no pyproject.toml found)")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_selected_images(path: Path) -> list[str]:
    """Read a newline-separated image list, ignoring blank and '#' lines."""
    filenames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            filenames.append(stripped)
    return filenames


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024
    return f"{num_bytes:.2f} GB"


@dataclass
class ImageInfo:
    """Result of inspecting one image file on disk."""

    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    sha256: str = ""
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.width is not None and self.height is not None


def inspect_image(path: Path) -> ImageInfo:
    """Open an image, fully decode it and return its metadata or an error."""
    if not path.is_file():
        return ImageInfo(error="file not found")
    try:
        digest = sha256_file(path)
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            image.load()
        return ImageInfo(width=width, height=height, image_format=image_format, sha256=digest)
    except UnidentifiedImageError:
        return ImageInfo(error="not a recognizable image file")
    except OSError as error:
        return ImageInfo(error=f"corrupted or unreadable image ({error})")


def write_manifest(path: Path, records: Iterable[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def read_manifest(path: Path) -> list[ImageRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [ImageRecord.from_row(row) for row in csv.DictReader(handle)]


def local_file_url_prefix(config: dict[str, Any]) -> str:
    """Label Studio URL prefix for raw images, derived from the config path.

    label_studio.sh (and the compose file) mount the host ``data/`` directory
    read-only at ``/label-studio/data/data``, so the URL path is the config
    path under ``data/`` itself. Deriving it here keeps the tasks and the
    mounts from drifting apart.
    """
    raw_images = PurePosixPath(str(config["paths"]["raw_images"]).strip("/"))
    return f"/data/local-files/?d={raw_images.as_posix()}/"


def write_label_studio_tasks(path: Path, records: Iterable[ImageRecord], url_prefix: str) -> int:
    """Write Label Studio tasks that reference raw images via local storage."""
    tasks = [
        {"id": record.image_id, "data": {"image": f"{url_prefix}{record.filename}"}}
        for record in records
    ]
    write_json(path, tasks)
    return len(tasks)


def load_coco(source: Path) -> AnnotationFile:
    """Load COCO JSON from a ``.json`` file or a Label Studio export ``.zip``."""
    if source.suffix.lower() == ".zip":
        payload = _read_coco_from_zip(source)
    else:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    return _parse_coco(payload, source)


def _read_coco_from_zip(source: Path) -> Any:
    with zipfile.ZipFile(source) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".json") and not name.startswith("__MACOSX")
        ]
        if len(candidates) != 1:
            raise CocoFormatError(
                f"{source}: expected exactly one JSON file in the archive, found {candidates}"
            )
        with archive.open(candidates[0]) as handle:
            return json.load(handle)


def _parse_coco(payload: Any, source: Path) -> AnnotationFile:
    if not isinstance(payload, dict):
        raise CocoFormatError(f"{source}: top-level JSON value must be an object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise CocoFormatError(f"{source}: missing or invalid '{key}' list")

    annotation_file = AnnotationFile()
    for image in payload["images"]:
        try:
            image_id = int(image["id"])
            filename = str(image["file_name"])
        except (KeyError, TypeError, ValueError) as error:
            raise CocoFormatError(f"{source}: invalid image entry: {image!r}") from error
        annotation_file.images[image_id] = ImageRecord(
            image_id=image_id,
            filename=Path(filename).name,
            width=int(image["width"]) if image.get("width") is not None else None,
            height=int(image["height"]) if image.get("height") is not None else None,
            status="ok",
        )
    for category in payload["categories"]:
        try:
            annotation_file.categories[int(category["id"])] = Category(
                id=int(category["id"]), name=str(category["name"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CocoFormatError(f"{source}: invalid category entry: {category!r}") from error
    for item in payload["annotations"]:
        try:
            annotation_file.annotations.append(Annotation.from_coco(item))
        except (KeyError, TypeError, ValueError) as error:
            raise CocoFormatError(f"{source}: invalid annotation entry: {item!r}") from error
    return annotation_file
