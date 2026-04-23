"""Batched closed-form propagator for a composite (phi, tau) pulse sequence under ORE+PLE."""

from __future__ import annotations

import math

import torch


def batched_unitary_generator(pulses: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Compose U_L ⋯ U_1 for a batch of (phi, tau) sequences under static (δ, ε) errors.

    Args:
        pulses: (B, L, 2) — each pulse is (phi, tau).
        error:  (2, B) — row 0 is off-resonant detuning δ, row 1 is pulse-length error ε.
                For no PLE, pass ε = zeros.
    Returns:
        (B, 2, 2) composite unitary.

    Hamiltonian per pulse: H = 0.5·(1+ε)·(cos φ X + sin φ Y + δ Z).
    Closed-form 2×2 propagator via Rodrigues avoids per-pulse matrix_exp.
    """
    if pulses.ndim != 3 or pulses.shape[-1] != 2:
        raise ValueError(f"'pulses' must have shape (B, L, 2); got {tuple(pulses.shape)}")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    phi, tau = pulses.unbind(dim=-1)           # each (B, L)
    delta = error[0]                           # (B,)
    epsilon = error[1]                         # (B,)

    A = 0.5 * (1.0 + epsilon[..., None]) * tau        # (B, L)
    v_x = A * torch.cos(phi)
    v_y = A * torch.sin(phi)
    v_z = A * delta[..., None]
    alpha = torch.sqrt(v_x * v_x + v_y * v_y + v_z * v_z)

    cos_a = torch.cos(alpha)
    sinc_a = torch.sinc(alpha / math.pi)              # sin(α)/α, safe at α=0

    u00 = (cos_a - 1j * sinc_a * v_z).to(dtype)
    u01 = (-sinc_a * (1j * v_x + v_y)).to(dtype)
    u10 = (-sinc_a * (1j * v_x - v_y)).to(dtype)
    u11 = (cos_a + 1j * sinc_a * v_z).to(dtype)

    U = torch.stack(
        [torch.stack([u00, u01], dim=-1),
         torch.stack([u10, u11], dim=-1)],
        dim=-2,
    )                                                  # (B, L, 2, 2)

    return _log_depth_product(U, B, device, dtype)


def _log_depth_product(U: torch.Tensor, B: int, device, dtype) -> torch.Tensor:
    """Left-to-right composite U_L ⋯ U_1 in O(log L) batched matmuls."""
    X = U
    I = torch.eye(2, dtype=dtype, device=device).expand(B, 1, 2, 2)
    while X.size(1) > 1:
        if X.size(1) & 1:
            X = torch.cat([X, I], dim=1)
        X = X[:, 1::2] @ X[:, 0::2]
    return X[:, 0]
