import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


__all__ = ["CompositePulseTransformerEncoder"]


###############################################################################
# Utility helpers – flattening unitaries and fidelity
###############################################################################

def _to_real_vector(U: torch.Tensor) -> torch.Tensor:
    """Flatten a complex matrix into a real‑valued vector (…, 2*d*d)."""
    real = U.real
    imag = U.imag
    return torch.cat([
        real.reshape(*real.shape[:-2], -1),
        imag.reshape(*imag.shape[:-2], -1)
    ], dim=-1)


def fidelity(U_out: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    """
    Entanglement fidelity F = |Tr(U_out† U_target)|² / d²
    Works for any batch size B and dimension d.
    """
    d = U_out.shape[-1]

    # Trace of U_out† U_target – no data movement thanks to index relabelling.
    inner = torch.einsum("bji,bij->b", U_out.conj(), U_target)  # shape [B]

    return (inner.conj() * inner / d ** 2).real                        # shape [B] or scalar


###############################################################################
# Model
###############################################################################

class CompositePulseTransformerEncoder(nn.Module):
    """Transformer encoder mapping *U_target* → pulse sequence.

    Each pulse is a continuous vector of parameters whose ranges are supplied
    via ``pulse_space``.  For a single‑qubit drive this could be (Δ, Ω, φ, t).
    """

    def __init__(
        self,
        num_qubits: int,
        pulse_space: Dict[str, Tuple[float, float]],
        max_pulses: int = 16,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        dropout: float = 0.1,
        score_emb: int = 4
    ) -> None:
        
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # ------------------------------------------------------------------
        # Pulse parameter space
        # ------------------------------------------------------------------
        self.param_names: Sequence[str] = list(pulse_space.keys())
        self.param_ranges: torch.Tensor = torch.tensor(
            [pulse_space[k] for k in self.param_names], dtype=torch.float32
        )  # (P, 2)
        self.param_dim = len(self.param_names)
        self.max_pulses = max_pulses
        self.d_model = d_model
        
        # Projection of flattened (real+imag) unitary → d_model
        self.unitary_proj = nn.Linear(score_emb * 2 * self.dim ** 2, d_model)

        # Transformer Encoder Model

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout
        )
        if n_layers is None:
            n_layers = 4 * max_pulses

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output linear head – maps encoder hidden → pulse parameters (normalised)
        self.head = nn.Linear(d_model, self.max_pulses * self.param_dim)



    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, 
        U_target: torch.Tensor
    ) -> torch.Tensor:
        """Generate pulses for every target unitary *U_target* (B, d, d)."""
    
        # Encode source (unitary) – shape (B, 1, d_model)
        emb = self.unitary_proj(_to_real_vector(U_target)).unsqueeze(1)

        logit = self.encoder(emb) # (B, 1, d_model)

        pulses_norm = self.head(logit)  # (B, 1, max_pulses * param_dim)

        # Map to physical ranges: sigmoid → (0,1) → (low, high)
        pulses_norm = pulses_norm.view(U_target.shape[0], self.max_pulses, self.param_dim)
        pulses_unit = pulses_norm.sigmoid()  # (B, L, P)
        low, high = self.param_ranges[:, 0].to(pulses_unit.device), self.param_ranges[:, 1].to(pulses_unit.device)
        pulses = low + (high - low) * pulses_unit
        return pulses