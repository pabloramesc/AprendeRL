"""PyTorch device selection."""

import torch


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""

    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
