from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .annotation_parsing import annotation_to_yolo_lines
from .constants import DEFAULT_DATASET_ROOT, DEFAULT_YOLO_ROOT


def image_path_to_export_name(image_path: str) -> str:
    path = Path(image_path)
    return f"{path.parent.name}-{path.name}"


def image_path_to_label_name(image_path: str) -> str:
    return Path(image_path_to_export_name(image_path)).with_suffix(".txt").name


def iter_dataset_images(dataset_root: Path) -> Iterable[Path]:
    images_root = dataset_root / "images"
    for book_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for image_path in sorted(book_dir.iterdir()):
            if image_path.is_file():
                yield image_path


def copy_images(dataset_root: Path, output_images_dir: Path):
    output_images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in iter_dataset_images(dataset_root):
        destination = output_images_dir / f"{image_path.parent.name}-{image_path.name}"
        shutil.copy2(image_path, destination)
        copied += 1
    return copied


def export_labels(metadata_path: Path, output_labels_dir: Path):
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(metadata_path)
    for _, row in df.iterrows():
        label_path = output_labels_dir / image_path_to_label_name(row["image_path"])
        yolo_lines = annotation_to_yolo_lines(row["image_annotation"])
        label_path.write_text("\n".join(yolo_lines))
    return len(df)


def build_manifest_lines(metadata_path: Path, image_prefix: str = "data/custom/images") -> List[str]:
    df = pd.read_pickle(metadata_path)
    return [f"{image_prefix}/{image_path_to_export_name(row['image_path'])}" for _, row in df.iterrows()]


def write_manifest(metadata_path: Path, output_manifest_path: Path, image_prefix: str = "data/custom/images"):
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = build_manifest_lines(metadata_path, image_prefix=image_prefix)
    output_manifest_path.write_text("\n".join(lines))
    return len(lines)


def prepare_yolo_split(metadata_path: Path, output_root: Path, manifest_name: str):
    labels_dir = output_root / "labels"
    export_labels(metadata_path, labels_dir)
    manifest_path = output_root / manifest_name
    write_manifest(metadata_path, manifest_path)
    return manifest_path


def default_custom_root() -> Path:
    return DEFAULT_YOLO_ROOT
