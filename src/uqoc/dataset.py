"""Sampling target SU(2) unitaries for training/evaluation."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from .quantum import rotation_vector_to_unitary


def build_SU2_dataset(size: int, random: bool = True,
                      device: torch.device | str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample target SU(2) gates.

    Args:
        size: Number of samples. If `random=False`, an (√size × √size) grid over
              (polar, azimuthal) is used; `size` must be a perfect square.
        random: If True, draw (θ, α, φ) uniformly; otherwise grid θ × φ, rand azimuth.
    Returns:
        rotation_vector : (size, 4) real tensor (n_x, n_y, n_z, θ)
        U_target        : (size, 2, 2) complex tensor
    """
    if random:
        theta = torch.rand(size) * math.pi
        alpha = torch.rand(size) * 2 * math.pi
        phi = torch.rand(size) * 2 * math.pi
    else:
        B = int(math.sqrt(size))
        if B * B != size:
            raise ValueError(f"grid mode needs a perfect square size; got {size}")
        theta_grid = torch.linspace(0, math.pi, B)
        phi_grid = torch.linspace(0, 2 * math.pi, B)
        theta, phi = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        theta = theta.flatten()
        phi = phi.flatten()
        alpha = torch.rand(B * B) * 2 * math.pi

    n_x = torch.sin(theta) * torch.cos(phi)
    n_y = torch.sin(theta) * torch.sin(phi)
    n_z = torch.cos(theta)
    n = torch.stack([n_x, n_y, n_z], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

    rotation_vector = torch.cat([n, alpha.unsqueeze(1)], dim=1).to(torch.float)
    U_target = rotation_vector_to_unitary(rotation_vector).to(torch.complex64)

    return rotation_vector.to(device), U_target.to(device)
