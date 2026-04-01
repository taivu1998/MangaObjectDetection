from pathlib import Path

LABELS = ("body", "face", "frame", "text")
NUM_LABELS = len(LABELS)

DETECTION_LABEL_MAP = {label: idx + 1 for idx, label in enumerate(LABELS)}
YOLO_LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

DEFAULT_DATASET_ROOT = Path("./Manga109")
DEFAULT_METADATA_ROOT = Path("./Manga109_metadata")
DEFAULT_YOLO_ROOT = Path("./YOLO/PyTorch-YOLOv3/data/custom")
DEFAULT_CHECKPOINT_ROOT = Path("./checkpoints")
DEFAULT_NST_OUTPUT_ROOT = Path("./Manga109/duplicate_images")
