"""Checkpoint helpers shared by PyTorch algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load tensor-only checkpoints across supported PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before the ``weights_only`` argument.
        return torch.load(path, map_location=device)
