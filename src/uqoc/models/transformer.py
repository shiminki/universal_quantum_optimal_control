"""Transformer encoder that consumes the SCORE-embedded target and emits pulse parameters.

Input : rotation vector (n_x, n_y, n_z, θ) of the target SU(2).
Flow  : reduce to (n_xy, 0, n_z, θ) + residual phi_0, Y-X-Y Euler decomposition,
        SCORE-embed each Euler angle into 3 unitaries → 9 tokens of dim 2×2.
        Flatten each token to reals, project to d_model, add sinusoidal PE, pass
        through a TransformerEncoder, then project the final token to
        `max_pulses × 2` pulse parameters.
Output: (B, max_pulses, 2) with columns (phi, tau).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantum import euler_yxy, score_sequence_from_yxy, to_real_vector
from .base import BaseController, pulse_range_tensor, register


@register("transformer")
class TransformerController(BaseController):
    def __init__(
        self,
        num_qubits: int,
        pulse_space: Dict[str, Tuple[float, float]],
        max_pulses: int = 16,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        dropout: float = 0.1,
        finetune: str | None = None,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.param_names = list(pulse_space.keys())
        self.param_ranges = pulse_range_tensor(pulse_space)  # plain attr, not state_dict
        self.param_dim = len(self.param_names)
        self.max_pulses = max_pulses
        self.d_model = d_model
        self.finetune = finetune
        if self.param_dim != 2:
            raise ValueError("TransformerController expects a 2-param pulse_space (phi, tau).")

        self.unitary_proj = nn.Linear(2 * self.dim ** 2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, max_pulses * self.param_dim)

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

        euler_angles = euler_yxy(rotation_vector_reduced)                 # (B, 3)
        score_seq = score_sequence_from_yxy(euler_angles)                 # (B, 9, 2, 2)
        tokens = to_real_vector(score_seq).to(torch.float).to(device)     # (B, 9, 8)

        emb = self.unitary_proj(tokens)                                    # (B, 9, d_model)
        emb = emb + _sinusoidal_pe(emb.shape[1], self.d_model, device).unsqueeze(0)

        hidden = self.encoder(emb)                                         # (B, 9, d_model)
        pulses_norm = self.head(hidden[:, -1, :])                          # (B, max_pulses*P)
        pulses_norm = pulses_norm.view(B, self.max_pulses, self.param_dim)

        low = self.param_ranges[:, 0].to(device)
        high = self.param_ranges[:, 1].to(device)
        pulses = low + (high - low) * pulses_norm.sigmoid()

        if self.finetune:
            base = torch.load(self.finetune, map_location=device).unsqueeze(0).expand(B, -1, -1)
            pulses = 0.2 * pulses + base

        pulses[:, :, -1] = F.relu(pulses[:, :, -1])
        pulses[:, :, 0] = pulses[:, :, 0] + phi_0.unsqueeze(1)
        pulses[:, :, 0] = (pulses[:, :, 0] + math.pi) % (2 * math.pi) - math.pi

        return pulses


def _sinusoidal_pe(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe
