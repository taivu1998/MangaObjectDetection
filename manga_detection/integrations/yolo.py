from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def run_command(command, cwd: Path | None = None):
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def run_yolo_train(*, runner_prefix, model, data, pretrained_weights, cwd: Path | None = None, extra_args=None):
    command = list(runner_prefix) + [
        "yolo-train",
        "--model",
        model,
        "--data",
        data,
        "--pretrained_weights",
        pretrained_weights,
    ]
    if extra_args:
        command.extend(extra_args)
    run_command(command, cwd=cwd)


def run_yolo_detect_external(
    *,
    runner_prefix,
    images,
    weights,
    model,
    classes,
    output_dir: Path | None = None,
    cwd: Path | None = None,
    extra_args=None,
):
    command = list(runner_prefix) + [
        "yolo-detect",
        "--images",
        images,
        "--weights",
        weights,
        "--model",
        model,
        "--classes",
        classes,
    ]
    if output_dir is not None:
        command.extend(["--output", str(output_dir)])
    if extra_args:
        command.extend(extra_args)
    run_command(command, cwd=cwd)


def _import_pytorchyolo():
    try:  # pragma: no cover - optional dependency
        import numpy as np
        import torch
        import tqdm
        import torchvision.transforms as transforms
        from torch.autograd import Variable
        from torch.utils.data import DataLoader

        from pytorchyolo.models import load_model
        from pytorchyolo.utils.datasets import ImageFolder
        from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS, Resize
        from pytorchyolo.utils.utils import non_max_suppression, rescale_boxes, to_cpu
    except ImportError as exc:  # pragma: no cover - exercised only with missing dependency
        raise RuntimeError(
            "Native YOLO inference requires the `pytorchyolo` package. "
            "Install it or use `--backend external` to delegate to the external CLI."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "tqdm": tqdm,
        "transforms": transforms,
        "Variable": Variable,
        "DataLoader": DataLoader,
        "load_model": load_model,
        "ImageFolder": ImageFolder,
        "DEFAULT_TRANSFORMS": DEFAULT_TRANSFORMS,
        "Resize": Resize,
        "non_max_suppression": non_max_suppression,
        "rescale_boxes": rescale_boxes,
        "to_cpu": to_cpu,
    }


def load_class_names(classes_path: Path) -> list[str]:
    return [line.strip() for line in Path(classes_path).read_text().splitlines() if line.strip()]


def iter_input_images(images_path: Path):
    if images_path.is_file():
        yield images_path
        return
    for candidate in sorted(images_path.iterdir()):
        if candidate.is_file():
            yield candidate


def _class_color(class_idx: int):
    palette = [
        (230, 57, 70),
        (69, 123, 157),
        (42, 157, 143),
        (244, 162, 97),
        (168, 218, 220),
        (233, 196, 106),
    ]
    return palette[class_idx % len(palette)]


def render_detections_to_image(
    *,
    image_path: Path,
    detections,
    class_names: list[str],
    output_path: Path,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if detections:
        for detection in detections:
            x1, y1, x2, y2, conf, class_idx = detection
            color = _class_color(int(class_idx))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            class_name = class_names[int(class_idx)] if 0 <= int(class_idx) < len(class_names) else str(class_idx)
            label = f"{class_name} {float(conf):.2f}"
            text_origin = (x1 + 2, max(0, y1 - 12))
            draw.text(text_origin, label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _create_data_loader(runtime, img_path: Path, batch_size: int, img_size: int, n_cpu: int):
    dataset = runtime["ImageFolder"](
        str(img_path),
        transform=runtime["transforms"].Compose([runtime["DEFAULT_TRANSFORMS"], runtime["Resize"](img_size)]),
    )
    return runtime["DataLoader"](
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
    )


def _normalize_detections(runtime, detections, img_size, image_shape):
    if detections is None:
        return []
    rescaled = runtime["rescale_boxes"](detections, img_size, image_shape)
    rows = runtime["to_cpu"](rescaled).tolist()
    normalized = []
    for row in rows:
        x1, y1, x2, y2, conf, class_idx = row
        normalized.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(class_idx)))
    return normalized


def detect_directory_native(
    *,
    model_path: Path,
    weights_path: Path,
    img_path: Path,
    classes_path: Path,
    output_dir: Path,
    predictions_json: Path | None = None,
    batch_size: int = 1,
    img_size: int = 416,
    n_cpu: int = 0,
    conf_thres: float = 0.5,
    nms_thres: float = 0.4,
):
    runtime = _import_pytorchyolo()
    torch = runtime["torch"]
    model = runtime["load_model"](str(model_path), str(weights_path))
    class_names = load_class_names(classes_path)
    dataloader = _create_data_loader(runtime, img_path, batch_size, img_size, n_cpu)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    saved_predictions = []
    for image_paths, input_images in runtime["tqdm"].tqdm(dataloader, desc="Detecting"):
        input_images = runtime["Variable"](input_images.type(tensor_type))
        with torch.no_grad():
            detections_batch = model(input_images)
            detections_batch = runtime["non_max_suppression"](detections_batch, conf_thres, nms_thres)

        for image_path_str, detections in zip(image_paths, detections_batch):
            image_path = Path(image_path_str)
            image_size = Image.open(image_path).convert("RGB").size
            normalized = _normalize_detections(runtime, detections, img_size, image_size[::-1])
            output_path = output_dir / f"{image_path.stem}.png"
            render_detections_to_image(
                image_path=image_path,
                detections=normalized,
                class_names=class_names,
                output_path=output_path,
            )
            saved_predictions.append(
                {
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                    "detections": [
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                            "class_idx": class_idx,
                            "class_name": class_names[class_idx] if 0 <= class_idx < len(class_names) else str(class_idx),
                        }
                        for x1, y1, x2, y2, confidence, class_idx in normalized
                    ],
                }
            )

    if predictions_json is not None:
        predictions_json.parent.mkdir(parents=True, exist_ok=True)
        predictions_json.write_text(json.dumps(saved_predictions, indent=2))

    return saved_predictions


def build_train_parser():
    parser = argparse.ArgumentParser(description="Run external YOLO training.")
    parser.add_argument("--cwd", type=Path, default=Path("./YOLO/PyTorch-YOLOv3"))
    parser.add_argument("--model", default="config/yolov3-awesomeconfig.cfg")
    parser.add_argument("--data", default="config/custom.data")
    parser.add_argument("--pretrained-weights", default="weights/darknet53.conv.74")
    parser.add_argument("--runner-prefix", nargs="*", default=["poetry", "run"])
    parser.add_argument("extra_args", nargs="*")
    return parser


def train_main(argv=None):
    args = build_train_parser().parse_args(argv)
    run_yolo_train(
        runner_prefix=args.runner_prefix,
        model=args.model,
        data=args.data,
        pretrained_weights=args.pretrained_weights,
        cwd=args.cwd,
        extra_args=args.extra_args,
    )


def build_detect_parser():
    parser = argparse.ArgumentParser(description="Run YOLO inference and save rendered detection images.")
    parser.add_argument("--backend", choices=("native", "external"), default="native")
    parser.add_argument("--cwd", type=Path, default=Path("./YOLO/PyTorch-YOLOv3"))
    parser.add_argument("--images", type=Path, default=Path("data/samples/"))
    parser.add_argument("--weights", type=Path, default=Path("checkpoints/yolov3_ckpt_7.pth"))
    parser.add_argument("--model", type=Path, default=Path("config/yolov3-awesomeconfig.cfg"))
    parser.add_argument("--classes", type=Path, default=Path("data/custom/classes.names"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--predictions-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--nms-thres", type=float, default=0.4)
    parser.add_argument("--runner-prefix", nargs="*", default=["poetry", "run"])
    parser.add_argument("extra_args", nargs="*")
    return parser


def detect_main(argv=None):
    args = build_detect_parser().parse_args(argv)
    if args.backend == "external":
        run_yolo_detect_external(
            runner_prefix=args.runner_prefix,
            images=str(args.images),
            weights=str(args.weights),
            model=str(args.model),
            classes=str(args.classes),
            output_dir=args.output_dir,
            cwd=args.cwd,
            extra_args=args.extra_args,
        )
        return

    detect_directory_native(
        model_path=args.cwd / args.model if not args.model.is_absolute() else args.model,
        weights_path=args.cwd / args.weights if not args.weights.is_absolute() else args.weights,
        img_path=args.cwd / args.images if not args.images.is_absolute() else args.images,
        classes_path=args.cwd / args.classes if not args.classes.is_absolute() else args.classes,
        output_dir=args.output_dir,
        predictions_json=args.predictions_json,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.num_workers,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
    )
