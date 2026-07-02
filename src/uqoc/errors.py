"""Static-error distributions: off-resonant detuning (ORE) and pulse-length error (PLE).

Samplers return a (2, B) tensor with row 0 = δ and row 1 = ε, matching the shape
`propagator.batched_unitary_generator` expects. δ is always dimensionless (δ/Ω);
the `*_mhz` samplers take physical MHz parameters and normalise by the Rabi
frequency Ω internally. A registry (`ERROR_SAMPLERS`) lets configs select a
sampler by name. All samplers accept `device=` so Monte-Carlo draws happen
directly on the GPU.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Sequence

import torch


ErrorSampler = Callable[..., torch.Tensor]

ERROR_SAMPLERS: Dict[str, ErrorSampler] = {}

# Inherent hBN defect detuning lines (MHz).
HBN_DETUNINGS_MHZ = (-96.0, -32.0, 32.0, 96.0)


def register(name: str) -> Callable[[ErrorSampler], ErrorSampler]:
    def _wrap(fn: ErrorSampler) -> ErrorSampler:
        ERROR_SAMPLERS[name] = fn
        return fn
    return _wrap


@register("ore")
def ore(batch_size: int, delta_std: float = 1.0, device=None) -> torch.Tensor:
    """Off-resonant detuning only. ε fixed to 0. Returns (2, B)."""
    delta = torch.randn(batch_size, device=device) * delta_std
    epsilon = torch.zeros(batch_size, device=device)
    return torch.stack([delta, epsilon])


@register("ore_ple")
def ore_ple(batch_size: int, delta_std: float = 1.0, epsilon_std: float = 0.05,
            device=None) -> torch.Tensor:
    """Joint ORE + PLE with independent Gaussian draws. Returns (2, B)."""
    delta = torch.randn(batch_size, device=device) * delta_std
    epsilon = torch.randn(batch_size, device=device) * epsilon_std
    return torch.stack([delta, epsilon])


@register("ore_mhz")
def ore_mhz(batch_size: int, delta_std_mhz: float = 90.0, omega_mhz: float = 30.0,
            epsilon_std: float = 0.0, device=None) -> torch.Tensor:
    """Physically parameterised ORE: δ ~ N(0, delta_std_mhz²) in MHz, normalised by Ω.

    For the Ω = 30 MHz hBN target, delta_std_mhz = 3Ω = 90 gives δ ~ N(0, 9Ω²),
    i.e. δ/Ω ~ N(0, 3²) as seen by the dimensionless propagator. Returns (2, B).
    """
    delta = torch.randn(batch_size, device=device) * (delta_std_mhz / omega_mhz)
    if epsilon_std > 0:
        epsilon = torch.randn(batch_size, device=device) * epsilon_std
    else:
        epsilon = torch.zeros(batch_size, device=device)
    return torch.stack([delta, epsilon])


@register("hbn_discrete")
def hbn_discrete(batch_size: int, detunings_mhz: Sequence[float] = HBN_DETUNINGS_MHZ,
                 omega_mhz: float = 30.0, epsilon_std: float = 0.0,
                 device=None) -> torch.Tensor:
    """Uniform draw over the discrete inherent hBN detuning lines. Returns (2, B).

    Meant for evaluation at the hardware's fixed δ ∈ {−96, −32, +32, +96} MHz.
    """
    lines = torch.tensor(detunings_mhz, dtype=torch.float32, device=device) / omega_mhz
    idx = torch.randint(len(detunings_mhz), (batch_size,), device=device)
    delta = lines[idx]
    if epsilon_std > 0:
        epsilon = torch.randn(batch_size, device=device) * epsilon_std
    else:
        epsilon = torch.zeros(batch_size, device=device)
    return torch.stack([delta, epsilon])


def make_sampler(name: str, params: dict, device=None,
                 omega_mhz: float | None = None) -> Callable[[int], torch.Tensor]:
    """Return `λ(batch_size) → errors` with `params` (and device/Ω, if accepted) bound.

    `omega_mhz` is injected only into samplers whose signature declares it and
    only when the curriculum params don't already override it, so dimensionless
    samplers (`ore`, `ore_ple`) keep working unchanged.
    """
    fn = ERROR_SAMPLERS[name]
    bound = dict(params)
    if omega_mhz is not None and "omega_mhz" not in bound \
            and "omega_mhz" in inspect.signature(fn).parameters:
        bound["omega_mhz"] = omega_mhz
    return lambda batch_size: fn(batch_size, device=device, **bound)
