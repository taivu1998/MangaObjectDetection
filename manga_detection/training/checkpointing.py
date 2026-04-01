from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.root_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.root_dir / "best.pt"

    def epoch_path(self, epoch: int) -> Path:
        return self.root_dir / f"epoch_{epoch:04d}.pt"

    def save(
        self,
        *,
        model,
        optimizer,
        epoch: int,
        metric: float,
        metric_name: str,
        config: dict,
        is_best: bool,
        save_epoch_copy: bool = True,
    ):
        payload = {
            "epoch": epoch,
            "metric": metric,
            "metric_name": metric_name,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, self.latest_path)
        if save_epoch_copy:
            torch.save(payload, self.epoch_path(epoch))
        if is_best:
            torch.save(payload, self.best_path)

    def load(self, model, optimizer=None, *, device="cpu", which="latest"):
        checkpoint_path = self.best_path if which == "best" else self.latest_path
        if not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload
