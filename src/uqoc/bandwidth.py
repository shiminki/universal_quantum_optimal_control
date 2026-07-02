"""Differentiable AWG bandwidth constraints for (phi, tau) pulse sequences.

The propagator works in dimensionless units: tau is the Rabi rotation angle in
radians, so physical time is t = tau / (2π·Ω) with Ω the Rabi frequency in MHz.
A physical cutoff f_c therefore maps to

    max phase slew   |dφ/dτ| ≤ f_c / Ω            (= 10 for f_c = 300, Ω = 30)
    Gaussian −3 dB σ  σ_τ = √(ln 2) · Ω / f_c      (radians of Rabi rotation)

Two interchangeable constraints, selected by config (`bandwidth.mode`):

* `GaussianLowPass`  — resamples each sequence onto a uniform oversampled time
  grid, smooths the unwrapped phase there with a Gaussian kernel matched to the
  cutoff (one grouped conv1d over the whole batch), and hands the band-limited
  fine-grid waveform (B, M, 2) to the propagator. Working in time rather than
  segment index keeps the cutoff exact for non-uniform segment durations, and
  filtering the phase keeps the constant-amplitude constraint that
  quadrature-wise filtering would break (see smoothing/smooth_pulse.py).
* `SlewRatePenalty`  — loss term penalising wrapped phase jumps between
  adjacent segments whose implied instantaneous frequency exceeds the cutoff.

Both are fully batched, differentiable, and NaN-free.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (−π, π], differentiable a.e. with unit gradient."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D phase unwrap along the last dim (segment index)."""
    d = wrap_angle(phi[..., 1:] - phi[..., :-1])
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


class GaussianLowPass(nn.Module):
    """Gaussian low-pass on the unwrapped pulse phase, matched to `cutoff_mhz`.

    Each sequence is resampled onto its own uniform time grid of M points
    (M ≥ oversample·L, raised further so σ_τ always spans ≥ 1 grid step), the
    unwrapped phase is filtered there with erf-integrated Gaussian bin weights
    (exact bin areas — no sub-sample aliasing), and the result is returned on
    the fine grid as (B, M, 2) with uniform per-sequence τ. Total duration is
    preserved differentiably. Grid geometry and kernel widths are detached so
    gradients flow through the phase values, not the discretisation.
    """

    def __init__(self, cutoff_mhz: float = 300.0, omega_mhz: float = 30.0,
                 max_radius: int = 32, oversample: int = 8) -> None:
        super().__init__()
        if cutoff_mhz <= 0 or omega_mhz <= 0:
            raise ValueError("cutoff_mhz and omega_mhz must be positive")
        if oversample < 1:
            raise ValueError("oversample must be >= 1")
        self.cutoff_mhz = cutoff_mhz
        self.omega_mhz = omega_mhz
        self.sigma_tau = math.sqrt(math.log(2.0)) * omega_mhz / cutoff_mhz
        self.max_radius = max_radius
        self.oversample = oversample

    def forward(self, pulses: torch.Tensor) -> torch.Tensor:
        """(B, L, 2) (phi, tau) → (B, M, 2) band-limited fine-grid waveform."""
        phi, tau = pulses.unbind(dim=-1)                       # each (B, L)
        B, L = phi.shape
        if L < 2:
            return pulses
        device, dtype = phi.device, phi.dtype

        T = tau.sum(dim=1)                                     # (B,) differentiable
        T_d = T.detach().clamp_min(1e-6)
        M = self.oversample * L
        M = max(M, int(math.ceil((T_d.max() / self.sigma_tau).item())))
        # cap the grid so 3·σ_idx_max fits inside max_radius — a truncated
        # Gaussian degrades into a boxcar with the wrong frequency response
        M_cap = int((self.max_radius / 3.0) * (T_d.min() / self.sigma_tau).item())
        M = max(min(M, M_cap), L)

        # piecewise-constant phase sampled at the grid midpoints of each sequence
        edges = tau.detach().cumsum(dim=1)                     # (B, L) right edges
        t_mid = (torch.arange(M, device=device, dtype=dtype) + 0.5) / M * T_d[:, None]
        idx = torch.searchsorted(edges.contiguous(), t_mid.contiguous()).clamp_max(L - 1)
        phi_grid = torch.gather(unwrap_phase(phi), 1, idx)     # (B, M)

        # erf-integrated Gaussian bin weights, per-sequence width (σ_idx ≥ 1)
        sigma_idx = self.sigma_tau * M / T_d                   # (B,)
        R = int(min(self.max_radius, max(1, math.ceil(3.0 * sigma_idx.max().item()))))
        j = torch.arange(-R, R + 1, device=device, dtype=dtype)
        s = sigma_idx[:, None] * math.sqrt(2.0)
        kernel = 0.5 * (torch.erf((j[None, :] + 0.5) / s) - torch.erf((j[None, :] - 0.5) / s))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)     # (B, 2R+1)

        x = F.pad(phi_grid.unsqueeze(0), (R, R), mode="replicate")     # (1, B, M+2R)
        phi_s = F.conv1d(x, kernel.unsqueeze(1), groups=B).squeeze(0)  # (B, M)

        tau_grid = (T / M)[:, None].expand(B, M)               # keeps Στ = T, differentiable
        return torch.stack([wrap_angle(phi_s), tau_grid], dim=-1)


class SlewRatePenalty(nn.Module):
    """Mean squared excess of the per-transition phase slew over the band limit.

    The wrapped phase jump Δφ between adjacent segments, spread over the
    midpoint-to-midpoint time (τ_j + τ_{j+1})/2, implies an instantaneous
    frequency |Δφ/Δt|/2π. Transitions within the f_c band are free; only the
    normalised excess is penalised, so the penalty is exactly zero for
    hardware-feasible pulses.
    """

    def __init__(self, cutoff_mhz: float = 300.0, omega_mhz: float = 30.0) -> None:
        super().__init__()
        if cutoff_mhz <= 0 or omega_mhz <= 0:
            raise ValueError("cutoff_mhz and omega_mhz must be positive")
        self.limit = cutoff_mhz / omega_mhz                    # max |dφ/dτ|

    def forward(self, pulses: torch.Tensor) -> torch.Tensor:
        """(B, L, 2) → scalar penalty."""
        phi, tau = pulses.unbind(dim=-1)
        if phi.shape[-1] < 2:
            return pulses.new_zeros(())
        dphi = wrap_angle(phi[:, 1:] - phi[:, :-1])
        dt = 0.5 * (tau[:, 1:] + tau[:, :-1]).clamp_min(1e-4)
        excess = F.relu(dphi.abs() / (dt * self.limit) - 1.0)
        return (excess ** 2).mean()


def make_bandwidth(mode: str, cutoff_mhz: float, omega_mhz: float,
                   max_radius: int = 32, oversample: int = 8,
                   ) -> Tuple[Optional[Callable], Optional[Callable]]:
    """Config → (pulse_transform | None, pulse_penalty | None).

    mode: 'none' | 'filter' | 'penalty' | 'both'. In 'both' mode the penalty
    sees the filtered fine-grid pulses, discouraging phase steps whose
    post-filter slew still exceeds the band limit.
    """
    if mode not in ("none", "filter", "penalty", "both"):
        raise ValueError(f"Unknown bandwidth mode '{mode}'")
    transform = GaussianLowPass(cutoff_mhz, omega_mhz, max_radius, oversample) \
        if mode in ("filter", "both") else None
    penalty = SlewRatePenalty(cutoff_mhz, omega_mhz) \
        if mode in ("penalty", "both") else None
    return transform, penalty
