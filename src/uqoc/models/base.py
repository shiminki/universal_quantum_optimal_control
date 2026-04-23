"""Controller base class and model registry.

All neural-network controllers take a rotation-vector representation of the
target SU(2) gate and emit a (B, L, 2) `(phi, tau)` pulse sequence.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


class BaseController(nn.Module):
    """Interface every controller must satisfy.

    Subclasses override `forward`. The attribute `num_qubits` is read by the
    trainer to pick the right fidelity normalisation.
    """

    num_qubits: int = 1

    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


Factory = Callable[..., BaseController]
MODEL_REGISTRY: Dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    def _wrap(cls: Factory) -> Factory:
        MODEL_REGISTRY[name] = cls
        return cls
    return _wrap


def build_model(cfg: Dict) -> BaseController:
    """`cfg` must carry a `type` key matching an entry in `MODEL_REGISTRY`."""
    cfg = dict(cfg)
    name = cfg.pop("type")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**cfg)


def pulse_range_tensor(pulse_space: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """(P, 2) tensor of (low, high) per parameter in pulse_space declaration order."""
    return torch.tensor([tuple(pulse_space[k]) for k in pulse_space], dtype=torch.float32)
