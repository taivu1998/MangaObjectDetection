from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

from .constants import DETECTION_LABEL_MAP, YOLO_LABEL_MAP


def _as_int(value) -> int:
    return int(float(value))


def _as_float(value) -> float:
    return float(value)


def _annotation_dimensions(annotation: Dict) -> Tuple[float, float]:
    return _as_float(annotation["@width"]), _as_float(annotation["@height"])


def is_valid_box(box: Dict) -> bool:
    xmin = _as_int(box["@xmin"])
    xmax = _as_int(box["@xmax"])
    ymin = _as_int(box["@ymin"])
    ymax = _as_int(box["@ymax"])
    return xmax > xmin and ymax > ymin


def filter_annotation_contents(annotation: Dict, allowed_labels: Optional[Iterable[str]] = None) -> Dict:
    """Return a copy of the annotation with invalid boxes removed."""
    allowed = set(allowed_labels) if allowed_labels is not None else None
    cleaned = {key: deepcopy(value) for key, value in annotation.items() if key != "contents"}
    cleaned_contents: List[Dict] = []
    for box in annotation.get("contents", []):
        if allowed is not None and box.get("type") not in allowed:
            continue
        if not is_valid_box(box):
            continue
        cleaned_contents.append(deepcopy(box))
    cleaned["contents"] = cleaned_contents
    return cleaned


def count_invalid_boxes(annotation: Dict) -> int:
    return sum(1 for box in annotation.get("contents", []) if not is_valid_box(box))


def _iter_parsed_objects(annotation: Dict, label_map: Dict[str, int]):
    for obj in annotation.get("contents", []):
        label = obj.get("type")
        if label not in label_map:
            continue
        if not is_valid_box(obj):
            continue
        xmin = _as_float(obj["@xmin"])
        xmax = _as_float(obj["@xmax"])
        ymin = _as_float(obj["@ymin"])
        ymax = _as_float(obj["@ymax"])
        yield label_map[label], (xmin, ymin, xmax, ymax)


def annotation_to_detection_target(annotation: Dict, image_id: Optional[int] = None, label_map=None):
    """Convert a Manga109 annotation into a TorchVision detection target."""
    import torch

    label_map = label_map or DETECTION_LABEL_MAP
    parsed = list(_iter_parsed_objects(annotation, label_map))
    if parsed:
        boxes = torch.tensor([box for _, box in parsed], dtype=torch.float32)
        labels = torch.tensor([label for label, _ in parsed], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        area = torch.zeros((0,), dtype=torch.float32)

    target = {
        "boxes": boxes,
        "labels": labels,
        "area": area,
        "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
    }
    if image_id is not None:
        target["image_id"] = torch.tensor([image_id], dtype=torch.int64)
    return target


def annotation_to_yolo_rows(annotation: Dict, label_map=None):
    """Convert a Manga109 annotation into YOLO-normalized rows."""
    label_map = label_map or YOLO_LABEL_MAP
    width, height = _annotation_dimensions(annotation)
    if width <= 0 or height <= 0:
        return []

    rows = []
    for label, (xmin, ymin, xmax, ymax) in _iter_parsed_objects(annotation, label_map):
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        rows.append((label, x_center, y_center, box_width, box_height))
    return rows


def annotation_to_yolo_lines(annotation: Dict, label_map=None) -> List[str]:
    return [
        f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for label, x_center, y_center, width, height in annotation_to_yolo_rows(annotation, label_map=label_map)
    ]
