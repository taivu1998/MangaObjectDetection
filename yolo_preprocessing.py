import argparse
from pathlib import Path

from manga_detection.data.constants import DEFAULT_DATASET_ROOT, DEFAULT_YOLO_ROOT
from manga_detection.data.yolo_export import copy_images


def build_parser():
    parser = argparse.ArgumentParser(description="Copy Manga109 images into the YOLO custom images directory.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-images-dir", type=Path, default=DEFAULT_YOLO_ROOT / "images")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    copied = copy_images(args.dataset_root, args.output_images_dir)
    print(f"Copied {copied} images into {args.output_images_dir.resolve()}")


if __name__ == "__main__":
    main()
