from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

from ..data.constants import DEFAULT_CHECKPOINT_ROOT, DEFAULT_METADATA_ROOT, NUM_LABELS
from ..data.datasets import Manga109DetectionDataset, build_transforms, default_split_paths
from .checkpointing import CheckpointManager
from .utils import collate_fn, set_seed


LOGGER = logging.getLogger("manga_detection.training")


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()})
    return moved


def build_model(model_name: str, pretrained: bool = True):
    num_classes = NUM_LABELS + 1
    if model_name == "faster_rcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if model_name == "retinanet":
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
        return torchvision.models.detection.retinanet_resnet50_fpn(
            num_classes=num_classes,
            weights=weights,
            weights_backbone=weights_backbone,
        )
    raise ValueError(f"Unsupported model_name={model_name}")


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate_loss(model, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += float(loss.item())
            num_batches += 1
    return total_loss / max(num_batches, 1)


def build_dataloaders(metadata_root: Path, batch_size: int, num_workers: int, family: str, augmentation: str, alternate_image_root=None):
    split_paths = default_split_paths(metadata_root, family=family)
    datasets = {
        "train": Manga109DetectionDataset(
            split_paths["train"],
            transforms=build_transforms(train=True, augmentation=augmentation),
            alternate_image_root=alternate_image_root,
        ),
        "valid": Manga109DetectionDataset(
            split_paths["valid"],
            transforms=build_transforms(train=False, augmentation=augmentation),
            alternate_image_root=alternate_image_root,
        ),
    }
    return {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "valid": torch.utils.data.DataLoader(
            datasets["valid"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }


def run_training(
    *,
    model_name: str,
    metadata_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    pretrained: bool,
    seed: int,
    family: str = "data_condensed",
    augmentation: str = "default",
    alternate_image_root: Path | None = None,
    resume: str | None = None,
):
    configure_logging()
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(
        metadata_root=metadata_root,
        batch_size=batch_size,
        num_workers=num_workers,
        family=family,
        augmentation=augmentation,
        alternate_image_root=alternate_image_root,
    )

    model = build_model(model_name, pretrained=pretrained)
    model.to(device)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)

    checkpoint_manager = CheckpointManager(output_dir)
    config = {
        "model_name": model_name,
        "metadata_root": str(metadata_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_workers": num_workers,
        "pretrained": pretrained,
        "seed": seed,
        "family": family,
        "augmentation": augmentation,
        "alternate_image_root": str(alternate_image_root) if alternate_image_root else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    start_epoch = 0
    best_metric = float("inf")
    if resume in {"latest", "best"}:
        payload = checkpoint_manager.load(model, optimizer, device=device, which=resume)
        if payload is not None:
            start_epoch = int(payload["epoch"])
            best_metric = float(payload.get("metric", best_metric))
            LOGGER.info("Resumed %s checkpoint from epoch %s", resume, start_epoch)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, optimizer, dataloaders["train"], device)
        val_loss = evaluate_loss(model, dataloaders["valid"], device)
        is_best = val_loss <= best_metric
        best_metric = min(best_metric, val_loss)
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metric=val_loss,
            metric_name="val_loss",
            config=config,
            is_best=is_best,
            save_epoch_copy=True,
        )
        LOGGER.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f best_val_loss=%.4f",
            epoch + 1,
            train_loss,
            val_loss,
            best_metric,
        )


def build_parser(default_model: str, default_epochs: int, default_batch_size: int, default_lr: float, default_output_dir: Path):
    parser = argparse.ArgumentParser(description=f"Train a {default_model} detector.")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--learning-rate", type=float, default=default_lr)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", default="data_condensed")
    parser.add_argument("--augmentation", choices=("default", "strong", "none"), default="default")
    parser.add_argument("--alternate-image-root", type=Path, default=None)
    parser.add_argument("--resume", choices=("latest", "best"), default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(
    argv=None,
    *,
    default_model: str,
    default_epochs: int,
    default_batch_size: int,
    default_lr: float,
    default_output_name: str,
    default_alternate_image_root: Path | None = None,
):
    parser = build_parser(
        default_model=default_model,
        default_epochs=default_epochs,
        default_batch_size=default_batch_size,
        default_lr=default_lr,
        default_output_dir=DEFAULT_CHECKPOINT_ROOT / default_output_name,
    )
    args = parser.parse_args(argv)
    alternate_root = args.alternate_image_root or default_alternate_image_root
    run_training(
        model_name=default_model,
        metadata_root=args.metadata_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        family=args.family,
        augmentation=args.augmentation,
        alternate_image_root=alternate_root,
        resume=args.resume,
    )


if __name__ == "__main__":  # pragma: no cover
    main(
        default_model="faster_rcnn",
        default_epochs=10,
        default_batch_size=4,
        default_lr=1e-4,
        default_output_name="faster-rcnn_pretrained_augment_lr=0.0001_imgaug",
    )
