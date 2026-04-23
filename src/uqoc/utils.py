"""Seeding and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(prefer: str | None = None) -> torch.device:
    """Pick cuda if available, else cpu. MPS is skipped because complex
    dtype (needed for unitary propagation) is unsupported there."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)
