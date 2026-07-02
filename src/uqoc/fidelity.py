"""Fidelity and loss functions.

`fidelity` is a thin re-export of `uqoc.quantum.fidelity`. Loss functions are
registered in `LOSS_REGISTRY` so configs can pick by name; `make_loss` binds
extra hyper-parameters (e.g. the rotation-error weight λ) from config.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F_nn

from .quantum import fidelity, rotation_angle_error

__all__ = ["fidelity", "LOSS_REGISTRY", "make_loss", "neg_log_loss",
           "infidelity_loss", "sharp_loss", "sharp_rotation_loss"]


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
    """Zero gradient at F=1, sharp gradient below τ. L = softplus(-k(F-τ))·(1-F).

    softplus(x) = log(1+exp(x)) computed without overflow for large -k(F-τ).
    """
    F = fidelity(U_out, U_target, num_qubits).mean()
    return F_nn.softplus(-k * (F - tau)) * (1.0 - F)


@register("sharp_rotation")
def sharp_rotation_loss(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int,
                        tau: float = 0.99, k: float = 100.0,
                        rotation_weight: float = 1.0) -> torch.Tensor:
    """sharp_loss(F) + λ·MSE(θ_target, θ_out) penalising rotation error directly.

    Rotation error ε = θ_out − θ_target diffuses as P_n = E_δ[|cos(n(π+ε)/2)|²]
    over repeated cycles, so it is penalised on top of the (phase-invariant)
    fidelity term. `rotation_angle_error` folds both angles onto [0, π], so an
    output on the −U_target lift (same physical gate, F = 1) incurs no penalty.
    """
    F = fidelity(U_out, U_target, num_qubits).mean()
    sharp = F_nn.softplus(-k * (F - tau)) * (1.0 - F)
    theta_err = rotation_angle_error(U_out, U_target)
    return sharp + rotation_weight * (theta_err ** 2).mean()


def make_loss(name: str, params: dict | None = None) -> LossFn:
    """Return `λ(U_out, U_target, num_qubits) → loss` with `params` bound."""
    fn = LOSS_REGISTRY[name]
    params = params or {}
    if not params:
        return fn
    return lambda U_out, U_target, num_qubits: fn(U_out, U_target, num_qubits, **params)
