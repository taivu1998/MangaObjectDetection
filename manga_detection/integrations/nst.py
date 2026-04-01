from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def iter_content_images(images_root: Path):
    for book_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for image_path in sorted(book_dir.iterdir()):
            if image_path.is_file():
                yield book_dir.name, image_path


def generate_nst_dataset(
    *,
    images_root: Path,
    output_root: Path,
    style_image: Path,
    model_path: Path,
    main_py: Path,
    cwd: Path | None = None,
    python_executable: str = sys.executable,
    content_size: int = 1024,
    cuda_device: str = "0",
    skip_existing: bool = True,
):
    output_root.mkdir(parents=True, exist_ok=True)
    generated = 0
    for book_name, image_path in iter_content_images(images_root):
        destination_dir = output_root / book_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / image_path.name
        if skip_existing and destination_path.exists():
            continue
        command = [
            python_executable,
            str(main_py),
            "eval",
            "--content-image",
            str(image_path),
            "--style-image",
            str(style_image),
            "--model",
            str(model_path),
            "--content-size",
            str(content_size),
            "--cuda",
            str(cuda_device),
            "--output-image",
            str(destination_path),
        ]
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)
        generated += 1
    return generated


def build_parser():
    parser = argparse.ArgumentParser(description="Generate NST-augmented Manga109 images using an external project.")
    parser.add_argument("--images-root", type=Path, default=Path("./Manga109/images"))
    parser.add_argument("--output-root", type=Path, default=Path("./Manga109/duplicate_images"))
    parser.add_argument("--style-image", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--main-py", type=Path, default=Path("main.py"))
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument("--content-size", type=int, default=1024)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generated = generate_nst_dataset(
        images_root=args.images_root,
        output_root=args.output_root,
        style_image=args.style_image,
        model_path=args.model_path,
        main_py=args.main_py,
        cwd=args.cwd,
        python_executable=args.python_executable,
        content_size=args.content_size,
        cuda_device=args.cuda_device,
        skip_existing=not args.no_skip_existing,
    )
    print(f"Generated {generated} stylized images into {args.output_root.resolve()}")
