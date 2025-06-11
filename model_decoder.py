import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F



class CompositePulseTransformerDecoder(nn.Module):
    """Transformer decoder mapping *U_targets* → pulse sequence.

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
        tokenizer: nn.Linear=None # (2 * self.dim ** 2, d_model)
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
        self.register_buffer("pos_ids", torch.arange(max_pulses + 1))
        self.pos_emb = nn.Embedding(max_pulses+1, d_model)


        # Tokenize target unitary
        if tokenizer is None:
            self.tokenizer = nn.Linear(2 * self.dim ** 2, d_model)
        else:
            self.tokenizer = tokenizer

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
        self.out = nn.Linear(d_model, self.param_dim)

        



    def _to_real_vector(self, U: torch.Tensor) -> torch.Tensor:
        real = U.real.reshape(U.shape[0], -1)
        imag = U.imag.reshape(U.shape[0], -1)
        return torch.cat([real, imag], dim=-1)

    def forward(self, U_targets: torch.Tensor) -> torch.Tensor:
        B = U_targets.shape[0]
        device = U_targets.device
        L = self.max_pulses + 1

        tokens = torch.zeros(B, L, self.d_model, device=device)
        src = self.tokenizer(self._to_real_vector(U_targets))
        tokens[:, 0] = src

        


        for i in range(self.max_pulses):
            tgt = tokens[:, :i + 1, :].clone()  # (B, i+1, d_model)
            # Position embedding
            pos = self.pos_emb(self.pos_ids[:i + 1]).unsqueeze(0).expand(B, -1, -1)
            tgt = tgt + pos

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(i + 1).to(device)

            logits = self.decoder(tgt, src.unsqueeze(1), tgt_mask=tgt_mask)
            # (B, i+1, d_model)

            tokens[:, i + 1] = logits[:, -1, :] # Shape: (B, d_model)

        pulses_logit = tokens[:, 1:] # (B, max_pulses, param_dim)
        pulses_norm = self.out(pulses_logit).sigmoid()
        low, high = self.param_ranges[:, 0].to(pulses_norm.device), self.param_ranges[:, 1].to(pulses_norm.device)
        pulses = low + (high - low) * pulses_norm
        return pulses

