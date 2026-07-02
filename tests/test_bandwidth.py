import math

import torch

from uqoc.bandwidth import GaussianLowPass, SlewRatePenalty, make_bandwidth, unwrap_phase


def _step_pulse(B: int = 4, L: int = 40, tau: float = 0.3) -> torch.Tensor:
    """Constant-τ sequence with a sharp π/2 phase jump in the middle."""
    phi = torch.zeros(B, L)
    phi[:, L // 2:] = math.pi / 2
    return torch.stack([phi, torch.full((B, L), tau)], dim=-1)


def _max_slew(pulses: torch.Tensor) -> float:
    """Max wrapped |Δφ| per unit τ across adjacent segments."""
    phi, tau = pulses.unbind(dim=-1)
    dphi = torch.atan2(torch.sin(phi.diff(dim=-1)), torch.cos(phi.diff(dim=-1)))
    dt = 0.5 * (tau[:, 1:] + tau[:, :-1])
    return (dphi.abs() / dt).max().item()


def test_unwrap_phase_removes_2pi_jumps():
    phi = torch.tensor([[3.0, -3.0, 3.0]])  # wraps across ±π
    u = unwrap_phase(phi)
    d = u[:, 1:] - u[:, :-1]
    assert d.abs().max() < math.pi


def test_lowpass_outputs_fine_grid_and_preserves_duration():
    pulses = _step_pulse(B=4, L=40)
    out = GaussianLowPass(cutoff_mhz=300.0, omega_mhz=30.0, oversample=8)(pulses)
    B, M, P = out.shape
    assert B == 4 and P == 2 and M >= 8 * 40
    assert torch.allclose(out[..., 1].sum(dim=-1), pulses[..., 1].sum(dim=-1), atol=1e-4)


def test_lowpass_enforces_band_at_operating_point():
    """π jump across short (τ=0.1) segments in a mostly-long sequence — the case
    the segment-index kernel missed. Post-filter slew must respect the ideal
    Gaussian step response (Δφ/(√(2π)σ_τ) ≈ 15.1), not the raw ≈ 31.4."""
    L = 50
    phi = torch.zeros(1, L)
    phi[:, L // 2:] = math.pi
    tau = torch.full((1, L), 0.5)
    tau[:, L // 2 - 2: L // 2 + 2] = 0.1
    pulses = torch.stack([phi, tau], dim=-1)
    assert _max_slew(pulses) > 30.0

    lp = GaussianLowPass(cutoff_mhz=300.0, omega_mhz=30.0, oversample=8)
    out = lp(pulses)
    ideal_step_slew = math.pi / (math.sqrt(2 * math.pi) * lp.sigma_tau)
    assert _max_slew(out) < 1.05 * ideal_step_slew


def test_lowpass_reduces_phase_jump():
    pulses = _step_pulse()
    out = GaussianLowPass(cutoff_mhz=60.0, omega_mhz=30.0)(pulses)
    jump_in = (pulses[..., 0].diff(dim=-1)).abs().max()
    jump_out = (out[..., 0].diff(dim=-1)).abs().max()
    assert jump_out < 0.5 * jump_in


def test_lowpass_is_differentiable():
    pulses = _step_pulse().requires_grad_(True)
    out = GaussianLowPass(cutoff_mhz=300.0, omega_mhz=30.0)(pulses)
    out.sum().backward()
    assert torch.isfinite(pulses.grad).all()


def test_slew_penalty_zero_within_band():
    # dφ/dτ limit is 300/30 = 10; a 0.5 rad jump over dt = 0.3 → rate ≈ 1.7, feasible.
    pulses = _step_pulse()
    pulses[..., 0] *= 0.5 / (math.pi / 2)
    penalty = SlewRatePenalty(cutoff_mhz=300.0, omega_mhz=30.0)(pulses)
    assert penalty.item() == 0.0


def test_slew_penalty_positive_beyond_band():
    # π jump over dt = 0.05 → rate ≈ 63 ≫ 10.
    pulses = _step_pulse(tau=0.05)
    penalty = SlewRatePenalty(cutoff_mhz=300.0, omega_mhz=30.0)(pulses)
    assert penalty.item() > 0.0


def test_slew_penalty_differentiable():
    pulses = _step_pulse(tau=0.05).requires_grad_(True)
    SlewRatePenalty(cutoff_mhz=300.0, omega_mhz=30.0)(pulses).backward()
    assert torch.isfinite(pulses.grad).all()


def test_make_bandwidth_modes():
    t, p = make_bandwidth("none", 300.0, 30.0)
    assert t is None and p is None
    t, p = make_bandwidth("filter", 300.0, 30.0)
    assert t is not None and p is None
    t, p = make_bandwidth("penalty", 300.0, 30.0)
    assert t is None and p is not None
    t, p = make_bandwidth("both", 300.0, 30.0)
    assert t is not None and p is not None
