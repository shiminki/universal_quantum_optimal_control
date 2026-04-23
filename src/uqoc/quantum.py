"""Pauli operators, SU(2) rotations, Euler Y-X-Y decomposition, SCORE composite pulses.

Single source of truth for the quantum-mechanical building blocks used by the
models (SCORE token embedding), propagator, dataset, and visualizations.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.func import vmap


_I2 = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)

_PAULI_CACHE: Dict[torch.device, torch.Tensor] = {}


def paulis(device: torch.device) -> torch.Tensor:
    """Return a (4, 2, 2) stack of (I, X, Y, Z) cached per device."""
    device = torch.device(device)
    if device not in _PAULI_CACHE:
        _PAULI_CACHE[device] = torch.stack([_I2, _SIGMA_X, _SIGMA_Y, _SIGMA_Z]).to(device)
    return _PAULI_CACHE[device]


def unit_vec(phi: torch.Tensor | float, dtype=torch.float64, device=None) -> torch.Tensor:
    """Unit vector (cos φ, sin φ, 0) in the x–y plane."""
    phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
    return torch.stack([torch.cos(phi_t), torch.sin(phi_t), torch.zeros_like(phi_t)], dim=-1)


def rotation_unitary(n: torch.Tensor, angle: torch.Tensor | float,
                     dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """Batched SU(2) rotation exp(-i θ/2 n·σ). n: (..., 3), angle: (...) → (..., 2, 2)."""
    n = torch.as_tensor(n)
    angle = torch.as_tensor(angle, device=n.device, dtype=n.dtype)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    c = torch.cos(angle / 2)
    s = -1j * torch.sin(angle / 2)
    row0 = torch.stack([c + s * z, s * (x - 1j * y)], dim=-1)
    row1 = torch.stack([s * (x + 1j * y), c - s * z], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(dtype)


def rotation_vector_to_unitary(rotation_vector: torch.Tensor) -> torch.Tensor:
    """(..., 4) rotation vector (n_x, n_y, n_z, θ) → (..., 2, 2) SU(2)."""
    n = rotation_vector[..., :3]
    theta = rotation_vector[..., 3]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return rotation_unitary(n, theta)


def to_real_vector(U: torch.Tensor) -> torch.Tensor:
    """Flatten a complex matrix (..., d, d) → real vector (..., 2*d*d), real/imag interleaved."""
    real = U.real.reshape(*U.shape[:-2], -1)
    imag = U.imag.reshape(*U.shape[:-2], -1)
    stacked = torch.stack((real, imag), dim=-1)
    return stacked.reshape(*U.shape[:-2], -1)


def euler_yxy(rotation_vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Vectorised Y-X-Y Euler decomposition.

    Args:
        rotation_vector: (..., 4) tensor (n_x, n_y, n_z, θ)
    Returns:
        (..., 3) tensor (α, β, γ) such that exp(-i θ/2 n·σ) = R_y(α)·R_x(β)·R_y(γ).
    """
    n, theta = rotation_vector[..., :3], rotation_vector[..., 3]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(eps)

    s, c = torch.sin(theta / 2), torch.cos(theta / 2)
    w, x, y, z = c, n[..., 0] * s, n[..., 1] * s, n[..., 2] * s

    beta = torch.acos((1.0 - 2.0 * (x ** 2 + z ** 2)).clamp(-1.0 + eps, 1.0 - eps))
    sin_beta = beta.sin()

    alpha_reg = torch.atan2(x * y - z * w, y * z + w * x)
    gamma_reg = torch.atan2(x * y + z * w, w * x - y * z)

    tol = 1e-6
    mask_reg = sin_beta.abs() > tol
    mask_beta0 = ~mask_reg & (beta < 0.5)
    mask_betapi = ~mask_reg & ~mask_beta0

    alpha = torch.empty_like(beta)
    gamma = torch.empty_like(beta)

    alpha[mask_reg] = alpha_reg[mask_reg]
    gamma[mask_reg] = gamma_reg[mask_reg]

    alpha[mask_beta0] = 2.0 * torch.atan2(y[mask_beta0], w[mask_beta0])
    gamma[mask_beta0] = 0.0

    alpha[mask_betapi] = 0.0
    gamma[mask_betapi] = 2.0 * torch.atan2(z[mask_betapi], x[mask_betapi])

    return torch.stack((alpha, beta, gamma), dim=-1)


def score_emb_unitary(phi: torch.Tensor | float, angle: torch.Tensor | float,
                      dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """SCORE composite implementing R_n(angle) for n = (cos φ, sin φ, 0). Returns (3, 2, 2)."""
    phi_t = torch.as_tensor(phi, dtype=torch.float64)
    angle_t = torch.as_tensor(angle, dtype=torch.float64)
    theta = math.pi - angle_t - torch.asin(0.5 * torch.sin(angle_t / 2))
    pi_t = torch.tensor(math.pi, dtype=torch.float64)
    return torch.stack([
        rotation_unitary(unit_vec(phi_t + pi_t), theta, dtype=dtype),
        rotation_unitary(unit_vec(phi_t), phi_t + 2 * theta, dtype=dtype),
        rotation_unitary(unit_vec(phi_t + pi_t), theta, dtype=dtype),
    ])


def score_sequence_from_yxy(euler_angles: torch.Tensor) -> torch.Tensor:
    """SCORE-embed a batch of Y-X-Y Euler triples into (B, 9, 2, 2) unitaries."""
    assert euler_angles.shape[-1] == 3

    def _single(angles: torch.Tensor) -> torch.Tensor:
        alpha, beta, gamma = angles.unbind()
        phis = torch.tensor([0.0, math.pi / 2, 0.0], dtype=angles.dtype, device=angles.device)
        thetas = torch.stack([alpha, beta, gamma])
        blocks = [score_emb_unitary(phi, theta) for phi, theta in zip(phis, thetas)]
        return torch.cat(blocks, dim=0)

    return vmap(_single)(euler_angles).to(euler_angles.device)


def fidelity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int = 1) -> torch.Tensor:
    """Entanglement fidelity F = (|Tr(U_out^† U_target)|² + d) / (d(d+1))."""
    product = U_out.conj().transpose(-1, -2) @ U_target
    trace = torch.einsum('...ii->...', product)
    d = 2 ** num_qubits
    return (trace.abs() ** 2 + d) / (d * (d + 1))
