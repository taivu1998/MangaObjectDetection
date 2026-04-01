import tempfile
import unittest
from pathlib import Path

import torch

from manga_detection.training.checkpointing import CheckpointManager


class CheckpointingTests(unittest.TestCase):
    def test_checkpoint_manager_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            manager.save(
                model=model,
                optimizer=optimizer,
                epoch=3,
                metric=1.5,
                metric_name="val_loss",
                config={"model": "toy"},
                is_best=True,
                save_epoch_copy=True,
            )

            loaded = manager.load(model, optimizer, device="cpu", which="best")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["epoch"], 3)
            self.assertEqual(loaded["metric_name"], "val_loss")
            self.assertTrue(manager.latest_path.exists())
            self.assertTrue(manager.best_path.exists())
            self.assertTrue(manager.epoch_path(3).exists())


if __name__ == "__main__":
    unittest.main()
