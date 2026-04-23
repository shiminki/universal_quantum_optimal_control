"""Dense MLP controller: rotation vector → pulse sequence.

Named `deep_nn` (not GRAPE) because training updates MLP weights, not pulse
parameters directly. The true GRAPE baseline lives under `baselines/`.

Architecture: six-layer MLP with ReLU on a 4-dim reduced target representation
(n_xy, 0, n_z, θ). Output head emits (ux, uy, τ) per pulse; phase is recovered
via atan2(uy, ux) to get a smooth angular output, τ is passed through sigmoid
then affinely mapped to the configured range.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseController, pulse_range_tensor, register


@register("deep_nn")
class DeepNNController(BaseController):
    def __init__(
        self,
        pulse_space: Dict[str, Tuple[float, float]],
        num_pulses: int,
        hidden_mult: int = 6,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.num_qubits = 1
        self.num_pulses = num_pulses
        self.param_names = list(pulse_space.keys())
        self.param_ranges = pulse_range_tensor(pulse_space)  # plain attr, not state_dict
        self.num_param = self.param_ranges.shape[0]
        if self.num_param != 2:
            raise ValueError("DeepNNController expects 2 pulse parameters (phi, tau).")

        hidden = num_pulses * hidden_mult
        layers: list[nn.Module] = [nn.Linear(4, hidden, bias=False), nn.ReLU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden, hidden, bias=False), nn.ReLU()]
        layers.append(nn.Linear(hidden, num_pulses * 3, bias=False))
        self.layer = nn.Sequential(*layers)

    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        B = rotation_vector.shape[0]
        device = rotation_vector.device

        phi_0 = torch.atan2(rotation_vector[:, 1], rotation_vector[:, 0])
        rotation_vector_reduced = torch.stack([
            torch.sqrt(rotation_vector[:, 0] ** 2 + rotation_vector[:, 1] ** 2),
            torch.zeros(B, device=device),
            rotation_vector[:, 2],
            rotation_vector[:, 3],
        ], dim=1)

        raw = self.layer(rotation_vector_reduced).reshape(B, self.num_pulses, 3)
        raw = raw.sigmoid()
        phi = torch.atan2(raw[:, :, 1], raw[:, :, 0])
        tau = raw[:, :, 2]
        pulses_unit = torch.stack((phi, tau), dim=-1)

        low = self.param_ranges[:, 0].to(device)
        high = self.param_ranges[:, 1].to(device)
        pulses = low + (high - low) * pulses_unit

        pulses[:, :, 0] = pulses[:, :, 0] + phi_0.unsqueeze(1)
        pulses[:, :, -1] = F.relu(pulses[:, :, -1])
        return pulses
