"""Static-error distributions: off-resonant detuning (ORE) and pulse-length error (PLE).

Samplers return a (2, B) tensor with row 0 = δ and row 1 = ε, matching the shape
`propagator.batched_unitary_generator` expects. A registry (`ERROR_SAMPLERS`) lets
configs select a sampler by name.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


ErrorSampler = Callable[..., torch.Tensor]

ERROR_SAMPLERS: Dict[str, ErrorSampler] = {}


def register(name: str) -> Callable[[ErrorSampler], ErrorSampler]:
    def _wrap(fn: ErrorSampler) -> ErrorSampler:
        ERROR_SAMPLERS[name] = fn
        return fn
    return _wrap


@register("ore")
def ore(batch_size: int, delta_std: float = 1.0) -> torch.Tensor:
    """Off-resonant detuning only. ε fixed to 0. Returns (2, B)."""
    delta = torch.randn(batch_size) * delta_std
    epsilon = torch.zeros(batch_size)
    return torch.stack([delta, epsilon])


@register("ore_ple")
def ore_ple(batch_size: int, delta_std: float = 1.0, epsilon_std: float = 0.05) -> torch.Tensor:
    """Joint ORE + PLE with independent Gaussian draws. Returns (2, B)."""
    delta = torch.randn(batch_size) * delta_std
    epsilon = torch.randn(batch_size) * epsilon_std
    return torch.stack([delta, epsilon])


def make_sampler(name: str, params: dict) -> Callable[[int], torch.Tensor]:
    """Return `λ(batch_size) → errors` with `params` bound."""
    fn = ERROR_SAMPLERS[name]
    return lambda batch_size: fn(batch_size, **params)
