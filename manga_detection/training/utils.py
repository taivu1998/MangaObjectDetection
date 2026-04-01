from __future__ import annotations

import random

import numpy as np
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
