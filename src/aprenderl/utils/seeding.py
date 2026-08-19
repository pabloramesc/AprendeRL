"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch when a seed is provided."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
