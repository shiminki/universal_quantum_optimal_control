import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


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

class CompositePulseTransformerDecoder(nn.Module):
    """Transformer decoder mapping *U_target* → pulse sequence.

    Each pulse is a continuous vector of parameters whose ranges are supplied
    via ``pulse_space``.  For a single‑qubit drive this could be (Δ, Ω, φ, t).
    """

    def __init__(
        self,
        num_qubits: int,
        pulse_space: Dict[str, Tuple[float, float]],
        *,
        max_pulses: int = 16,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        dropout: float = 0.1,
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

        # Positional embedding for pulse index (0 … max_pulses‑1)
        self.register_buffer("pos_ids", torch.arange(max_pulses))
        self.pos_emb = nn.Embedding(max_pulses, d_model)

        # Projection of flattened (real+imag) unitary → d_model
        self.unitary_proj = nn.Linear(2 * self.dim ** 2, d_model)

        # Learned start‑of‑sequence token
        self.sos = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer decoder (only the *decoder* – target‑side causal mask)
        layer = nn.TransformerDecoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        # Output linear head – maps decoder hidden → pulse parameters (normalised)
        self.head = nn.Linear(d_model, self.param_dim)


    @staticmethod
    def _bool_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=bool, device=device), diagonal=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, U_target: torch.Tensor, *, seq_len: int | None = None) -> torch.Tensor:
        """Generate *seq_len* pulses for every target unitary *U_target* (B, d, d)."""
        if seq_len is None:
            seq_len = self.max_pulses
        if seq_len > self.max_pulses:
            raise ValueError("seq_len exceeds max_pulses")

        B = U_target.shape[0]

        # Encode source (unitary) – shape (B, 1, d_model)
        src = self.unitary_proj(_to_real_vector(U_target)).unsqueeze(1)

        # Target side: [SOS, 0, 0, …] + positional encodings
        tgt_init = torch.zeros(B, seq_len, self.d_model, device=U_target.device)
        tgt_init[:, 0:1, :] = self.sos
        pos = self.pos_emb(self.pos_ids[:seq_len]).unsqueeze(0).expand(B, -1, -1)
        tgt = tgt_init + pos

        # Causal mask so pulse *k* cannot peek at future pulses
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(U_target.device)

        H = self.decoder(tgt, src, tgt_mask=tgt_mask)  # (B, seq_len, d_model)
        pulses_norm = self.head(H)  # values in ℝ

        # Map to physical ranges: sigmoid → (0,1) → (low, high)
        pulses_unit = pulses_norm.sigmoid()  # (B, L, P)
        low, high = self.param_ranges[:, 0].to(pulses_unit.device), self.param_ranges[:, 1].to(pulses_unit.device)
        pulses = low + (high - low) * pulses_unit
        return pulses