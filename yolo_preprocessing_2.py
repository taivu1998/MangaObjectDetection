import argparse
from pathlib import Path

from manga_detection.data.constants import DEFAULT_METADATA_ROOT, DEFAULT_YOLO_ROOT, LABELS
from manga_detection.data.yolo_export import prepare_yolo_split


def build_parser():
    parser = argparse.ArgumentParser(description="Export YOLO labels and split manifests from metadata pickle files.")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--family", default="data_condensed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    classes_path = args.output_root / "classes.names"
    classes_path.write_text("\n".join(LABELS))

    split_to_manifest = {
        "train": "train.txt",
        "valid": "valid.txt",
        "test": "test.txt",
    }
    for split, manifest_name in split_to_manifest.items():
        metadata_path = args.metadata_root / f"{args.family}_{split}.pkl"
        if not metadata_path.exists():
            continue
        manifest_path = prepare_yolo_split(metadata_path, args.output_root, manifest_name)
        print(f"Prepared {split} split from {metadata_path} -> {manifest_path}")


if __name__ == "__main__":
    main()
