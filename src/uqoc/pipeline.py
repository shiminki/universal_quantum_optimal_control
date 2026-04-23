"""Inference wrapper for a trained controller.

Accepts either a rotation-vector (n_x, n_y, n_z, θ) batch or a batch of target
unitaries (2, 2).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .models.base import BaseController


class Pipeline:
    def __init__(self, model: BaseController, weight_path: str | Path | None = None,
                 device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if weight_path is not None:
            state = torch.load(str(weight_path), map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        return self.model(rotation_vector.to(self.device))

    __call__ = forward

    @torch.no_grad()
    def forward_with_unitary(self, U_target: torch.Tensor) -> torch.Tensor:
        """Recover (n_x, n_y, n_z, θ) from each SU(2) and forward through the model."""
        U = U_target.to(self.device)
        n_x = torch.real(U[..., 0, 1])
        n_y = torch.imag(U[..., 0, 1])
        n_z = torch.real(U[..., 1, 1])
        theta = torch.acos(torch.real(U[..., 0, 0]).clamp(-1.0, 1.0))
        rotation_vector = torch.stack([n_x, n_y, n_z, theta], dim=-1)
        return self.forward(rotation_vector)
