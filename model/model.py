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
    """Flatten a complex matrix into a real‑valued vector with alternating real and imag components (…, 2*d*d)."""
    real = U.real.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)
    imag = U.imag.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)

    stacked = torch.stack((real, imag), dim=-1)  # shape: (..., d*d, 2)
    interleaved = stacked.reshape(*U.shape[:-2], -1)  # shape: (..., 2*d*d)

    return interleaved



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
# Positional Embedding
###############################################################################


def sinusoidal_positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding.

    Args:
        length: sequence length (e.g., max_pulses)
        d_model: embedding dimension
        device: torch device

    Returns:
        Tensor of shape (length, d_model)
    """
    position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

    pe = torch.zeros(length, d_model, device=device)  # (L, D)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # (L, D)


###############################################################################
# Model with SCORE Embedding
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
        finetune=False
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
        self.unitary_proj = nn.Linear(2 * self.dim ** 2, d_model)

        # Transformer Encoder Model

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        if n_layers is None:
            n_layers = 4 * max_pulses

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output linear head – maps encoder hidden → pulse parameters (normalised)
        self.head = nn.Linear(d_model, self.max_pulses * self.param_dim)

        # Whether to fine-tune from a base pulse
        self.finetune = finetune


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, 
        U_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate pulses for every target unitary *U_target* (B, L, d, d).
        L represents the length of the SCORE composite pulse
        """

        B = U_target.shape[0]
        L = U_target.shape[1] # Length of SCORE pulse
        D = self.d_model

        # Encode source (unitary) – shape (B, d_model)
        emb = self.unitary_proj(_to_real_vector(U_target))  # (B, L, D)

        # Add sinusoidal positional encoding (L, D) → broadcast to (B, L, D)
        pos_emb = sinusoidal_positional_encoding(L, D, device=emb.device)

        emb = emb + pos_emb.unsqueeze(0)  # (B, L, D)

        # Pass through transformer encoder
        logit = self.encoder(emb)  # (B, L, D)
        logit = self.head(logit)  # (B, L, P)

        pulses_norm = logit[:, -1, :]


        # Reshape to (B, L, max_pulse, num_param)
        pulses_norm = pulses_norm.view(B, self.max_pulses, self.param_dim)

        # Map to physical parameter range
        pulses_unit = pulses_norm.sigmoid()
        low = self.param_ranges[:, 0].to(pulses_unit.device)
        high = self.param_ranges[:, 1].to(pulses_unit.device)

        pulses = low + (high - low) * pulses_unit  # (B, L, P)

        if self.finetune:
            base_pulse = torch.load(self.finetune).to(pulses.device)
            base_pulse = base_pulse.unsqueeze(0).expand(B, -1, -1)
            pulses = 0.2 * pulses + base_pulse
            
        pulses[:, :, -1] = F.relu(pulses[:, :, -1])

        return pulses
    
 