from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tv_transforms
from PIL import Image

from .annotation_parsing import annotation_to_detection_target
from .constants import DEFAULT_METADATA_ROOT
from ..training.transforms import Compose


def _import_imgaug():
    try:  # pragma: no cover - optional dependency path
        import imgaug.augmenters as iaa  # type: ignore
        from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage  # type: ignore

        return iaa, BoundingBox, BoundingBoxesOnImage
    except Exception as exc:  # pragma: no cover - exercised only when optional dep missing/broken
        raise RuntimeError(
            "imgaug is not available or is incompatible with the current environment. "
            "Use --augmentation none or install a compatible imgaug stack."
        ) from exc


def _recompute_box_fields(target):
    boxes = target["boxes"]
    if boxes.numel() == 0:
        target["boxes"] = boxes.reshape(0, 4)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        target["labels"] = target["labels"].reshape(0)
    else:
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    return target


class ImgAugTransform:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, target):
        _, BoundingBox, BoundingBoxesOnImage = _import_imgaug()

        boxes = target["boxes"]
        labels = target["labels"]
        bbox_list = [
            BoundingBox(x1=box[0].item(), y1=box[1].item(), x2=box[2].item(), y2=box[3].item()) for box in boxes
        ]
        bounding_boxes = BoundingBoxesOnImage(bbox_list, shape=image.shape)
        image, bounding_boxes = self.augmentations(image=image, bounding_boxes=bounding_boxes)
        bounding_boxes = bounding_boxes.clip_out_of_image()

        kept_boxes = []
        kept_labels = []
        for idx, box in enumerate(bounding_boxes):
            xmin, ymin, xmax, ymax = box.x1, box.y1, box.x2, box.y2
            if xmax <= xmin or ymax <= ymin:
                continue
            kept_boxes.append([xmin, ymin, xmax, ymax])
            kept_labels.append(labels[idx].item())

        if kept_boxes:
            target["boxes"] = torch.tensor(kept_boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(kept_labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        return image, _recompute_box_fields(target)


class PadSquare(ImgAugTransform):
    def __init__(self):
        iaa, _, _ = _import_imgaug()
        super().__init__(iaa.Sequential([iaa.PadToAspectRatio(1.0, position="center-center")]))


class DefaultAug(ImgAugTransform):
    def __init__(self):
        iaa, _, _ = _import_imgaug()
        super().__init__(
            iaa.Sequential(
                [
                    iaa.Sharpen((0.0, 0.1)),
                    iaa.Affine(rotate=(0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                    iaa.AddToBrightness((-60, 40)),
                    iaa.AddToHue((-10, 10)),
                    iaa.Fliplr(0.5),
                ]
            )
        )


class StrongAug(ImgAugTransform):
    def __init__(self):
        iaa, _, _ = _import_imgaug()
        super().__init__(
            iaa.Sequential(
                [
                    iaa.Dropout([0.0, 0.01]),
                    iaa.Sharpen((0.0, 0.1)),
                    iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                    iaa.AddToBrightness((-60, 40)),
                    iaa.AddToHue((-20, 20)),
                    iaa.Fliplr(0.5),
                ]
            )
        )


class ToTensorTransform:
    def __call__(self, image, target):
        return tv_transforms.ToTensor()(image), target


def build_transforms(train: bool, augmentation: str = "default"):
    transforms = []
    if train and augmentation == "default":
        transforms.append(DefaultAug())
    elif train and augmentation == "strong":
        transforms.append(StrongAug())
    if augmentation != "none":
        transforms.append(PadSquare())
    transforms.append(ToTensorTransform())
    return Compose(transforms)


def resolve_alternate_image_path(image_path: str, alternate_image_root: Path | None) -> Path:
    path = Path(image_path)
    if alternate_image_root is None:
        return path
    parts = list(path.parts)
    if "images" in parts:
        start = parts.index("images") + 1
        relative = Path(*parts[start:])
        return alternate_image_root / relative
    return alternate_image_root / path.name


class Manga109DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path: Path, transforms=None, alternate_image_root: Path | None = None):
        self.metadata_path = Path(metadata_path)
        self.metadata = pd.read_pickle(self.metadata_path)
        self.transforms = transforms
        self.alternate_image_root = alternate_image_root

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = resolve_alternate_image_path(row["image_path"], self.alternate_image_root)
        image = np.array(Image.open(image_path).convert("RGB"))
        target = annotation_to_detection_target(row["image_annotation"], image_id=idx)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            target = _recompute_box_fields(target)
        return image, target


def split_metadata_path(metadata_root: Path, family: str, split: str) -> Path:
    return metadata_root / f"{family}_{split}.pkl"


def default_split_paths(metadata_root: Path | None = None, family: str = "data_condensed"):
    root = metadata_root or DEFAULT_METADATA_ROOT
    return {
        "train": split_metadata_path(root, family, "train"),
        "valid": split_metadata_path(root, family, "valid"),
        "test": split_metadata_path(root, family, "test"),
    }
