"""Fidelity and loss functions.

`fidelity` is a thin re-export of `uqoc.quantum.fidelity`. Loss functions are
registered in `LOSS_REGISTRY` so configs can pick by name.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from .quantum import fidelity

__all__ = ["fidelity", "LOSS_REGISTRY", "neg_log_loss", "infidelity_loss", "sharp_loss"]


LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
LOSS_REGISTRY: Dict[str, LossFn] = {}


def register(name: str) -> Callable[[LossFn], LossFn]:
    def _wrap(fn: LossFn) -> LossFn:
        LOSS_REGISTRY[name] = fn
        return fn
    return _wrap


@register("neg_log")
def neg_log_loss(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int) -> torch.Tensor:
    return -torch.log(fidelity(U_out, U_target, num_qubits).mean())


@register("infidelity")
def infidelity_loss(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int) -> torch.Tensor:
    return 1.0 - fidelity(U_out, U_target, num_qubits).mean()


@register("sharp")
def sharp_loss(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int,
               tau: float = 0.99, k: float = 100.0) -> torch.Tensor:
    """Zero gradient at F=1, sharp gradient below τ. L = log(1+exp(-k(F-τ)))·(1-F)."""
    F = fidelity(U_out, U_target, num_qubits).mean()
    return torch.log1p(torch.exp(-k * (F - tau))) * (1.0 - F)
