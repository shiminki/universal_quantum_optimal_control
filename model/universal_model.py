import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap   # <-- 1-line opt-in for vectorised maps

from tqdm import tqdm


__all__ = ["CompositePulseTransformerEncoder"]



###############################################################################
# Model with SCORE Embedding
###############################################################################

class UniversalQOCTransformer(nn.Module):
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
        rotation_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        rotation_vector: shape (B, 4) – target rotation axis and angle in the form of (n_x, n_y, n_z, theta).
        """
        B = rotation_vector.shape[0]
        L = 9  # SCORE sequence length
        D = self.d_model

        phi = torch.atan2(rotation_vector[:, 1], rotation_vector[:, 0])  # atan2(n_y, n_x)

        # rotation_vector_rescaled: shape (B, 4) – target rotation axis and angle in the form of (n_xy, 0, n_z, theta).
        rotation_vector_rescaled = torch.stack([
            torch.sqrt(rotation_vector[:, 0] ** 2 + rotation_vector[:, 1] ** 2),  # n_xy
            torch.zeros(B, device=rotation_vector.device),  # n_y
            rotation_vector[:, 2],  # n_z
            rotation_vector[:, 3]  # theta 
        ], dim=1)
        
        # YXY decomposition
        euler_angles = UniversalQOCTransformer.euler_yxy_from_rotation_vector(rotation_vector_rescaled)  # (B, 3)
        score_sequence = UniversalQOCTransformer.score_sequence_from_yxy(euler_angles) # (B, 9, 2, 2)

        # Flatten the score sequence to (B, 9*2*2)
        score_flat = UniversalQOCTransformer._to_real_vector(score_sequence).to(torch.float)  # (B, 9, 2*2)
        
        # Project to d_model
        emb = self.unitary_proj(score_flat.to(rotation_vector.device))  # (B, d_model)
    
        pos_emb = UniversalQOCTransformer.sinusoidal_positional_encoding(L, D, device=emb.device)

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
        pulses[:, :, 0] += phi.unsqueeze(1)

        return pulses


    @staticmethod
    def euler_yxy_from_rotation_vector(rotation_vector: torch.Tensor,
                                    eps: float = 1e-12) -> torch.Tensor:
        """
        Vectorised Y-X-Y Euler decomposition.
        Args
        ----
            rotation_vector : (B,4) tensor (n_x, n_y, n_z, θ)
        Returns
        -------
            (B,3) tensor (α, β, γ) such that
                exp(-i θ/2 n·σ) = R_y(α) · R_x(β) · R_y(γ)
        """
        n, theta = rotation_vector[..., :3], rotation_vector[..., 3]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(eps)       # normalise axis

        s, c = torch.sin(theta / 2), torch.cos(theta / 2)          # sin, cos θ/2
        w, x, y, z = c, n[..., 0] * s, n[..., 1] * s, n[..., 2] * s

        # ----- regular branch -------------------------------------------------- #
        beta = torch.acos((1.0 - 2.0 * (x**2 + z**2)).clamp(-1.0 + eps, 1.0 - eps))
        sin_beta = beta.sin()

        alpha_reg = torch.atan2(x * y - z * w, y * z + w * x)
        gamma_reg = torch.atan2(x * y + z * w, w * x - y * z)

        # ----- gimbal-lock handling ------------------------------------------- #
        tol = 1e-6
        mask_reg  = sin_beta.abs() > tol          # “normal” points
        mask_beta0 = ~mask_reg & (beta < 0.5)     # β ≈ 0         → Y-only
        mask_betapi = ~mask_reg & ~mask_beta0     # β ≈ π         → X/Z

        alpha = torch.empty_like(beta)
        gamma = torch.empty_like(beta)

        # regular
        alpha[mask_reg]  = alpha_reg[mask_reg]
        gamma[mask_reg]  = gamma_reg[mask_reg]

        # β ≈ 0  (rotation about Y)
        alpha[mask_beta0]  = 2.0 * torch.atan2(y[mask_beta0], w[mask_beta0])
        gamma[mask_beta0]  = 0.0

        # β ≈ π  (rotation about X or Z)
        alpha[mask_betapi] = 0.0
        gamma[mask_betapi] = 2.0 * torch.atan2(z[mask_betapi], x[mask_betapi])

        return torch.stack((alpha, beta, gamma), dim=-1)


    # ------------------------------------------------------------------ #
    # Low-level utilities                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def unit_vec(phi: float,
                 dtype: torch.dtype = torch.float64,
                 device=None) -> torch.Tensor:
        """Unit vector (cos φ, sin φ, 0) lying in the x–y plane."""
        return torch.tensor([math.cos(phi), torch.sin(phi), 0.0],
                            dtype=dtype, device=device)

    @staticmethod
    def rotation_unitary(n: torch.Tensor,
                         angle: float,
                         dtype: torch.dtype = torch.complex128) -> torch.Tensor:
        """
        Returns the unitary for rotation about axis n by angle.
        Supports batched n and angle.
        n: (..., 3)
        angle: (...,)
        Returns: (..., 2, 2)
        """
        # Ensure n and angle are tensors
        n = torch.as_tensor(n)
        angle = torch.as_tensor(angle, device=n.device, dtype=n.dtype)
        # Broadcast
        x, y, z = n[..., 0], n[..., 1], n[..., 2]
        c = torch.cos(angle / 2)
        s = -1j * torch.sin(angle / 2)
        # Build the matrix using stack and cat for batch support
        row0 = torch.stack([c + s * z, s * (x - 1j * y)], dim=-1)
        row1 = torch.stack([s * (x + 1j * y), c - s * z], dim=-1)
        U = torch.stack([row0, row1], dim=-2)
        return U.to(dtype)
    
    # ------------------------------------------------------------------ #
    # SCORE composite pulse                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_score_emb_unitary(phi: float,
                              angle: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            score_tensor  – (3,2,2) composite implementing the SCORE pulse
            target_unitary – (2,2) ideal rotation R_{n=unit_vec(φ)}(θ)
        """
        theta = torch.pi - angle - torch.asin(0.5 * torch.sin(angle / 2))

        pulses: List[torch.Tensor] = [
            UniversalQOCTransformer.rotation_unitary(
                UniversalQOCTransformer.unit_vec(phi + torch.pi), theta),
            UniversalQOCTransformer.rotation_unitary(
                UniversalQOCTransformer.unit_vec(phi), phi + 2 * theta),
            UniversalQOCTransformer.rotation_unitary(
                UniversalQOCTransformer.unit_vec(phi + torch.pi), theta)
        ]

        score_tensor = torch.stack(pulses)                       # (3,2,2)
        target_unitary = UniversalQOCTransformer.rotation_unitary(
            UniversalQOCTransformer.unit_vec(phi), angle)

        return score_tensor, target_unitary


    # ------------------------------------------------------------------ #
    # Build (B,9,2,2) tensor from Y-X-Y angles                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_sequence_from_yxy(euler_angles: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        euler_angles : (B, 3) real tensor of Y-X-Y Euler triples (α, β, γ)

        Returns
        -------
        (B, 9, 2, 2) complex tensor whose 9 unitaries are
        [ SCORE(0, α) • SCORE(π/2, β) • SCORE(0, γ) ]  (three pulses each)
        """
        assert euler_angles.shape[-1] == 3, "expected (..., 3) Euler input"

        # ---------- Inner helper: one sample -> nine unitaries ---------- #
        def _single_sequence(angles: torch.Tensor) -> torch.Tensor:
            alpha, beta, gamma = angles.unbind()
            phis   = torch.tensor([0.0, math.pi / 2, 0.0], dtype=angles.dtype, device=angles.device)
            thetas = torch.stack([alpha, beta, gamma])  # Use stack instead of tensor()

            blocks = [
                UniversalQOCTransformer.get_score_emb_unitary(phi, theta)[0]
                for phi, theta in zip(phis, thetas)
            ]
            return torch.cat(blocks, dim=0)  # (9,2,2)                           # (9,2,2)

        # vmap lifts the single-sample function to operate on the whole batch
        return vmap(_single_sequence)(euler_angles).to(euler_angles.device)          # (B,9,2,2)
                    # (B,9,2,2)


    ###############################################################################
    # Utility helpers – flattening unitaries and fidelity
    ###############################################################################
    @staticmethod
    def _to_real_vector(U: torch.Tensor) -> torch.Tensor:
        """Flatten a complex matrix into a real‑valued vector with alternating real and imag components (…, 2*d*d)."""
        real = U.real.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)
        imag = U.imag.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)

        stacked = torch.stack((real, imag), dim=-1)  # shape: (..., d*d, 2)
        interleaved = stacked.reshape(*U.shape[:-2], -1)  # shape: (..., 2*d*d)

        return interleaved

    @staticmethod
    def fidelity(U_out: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
        """
        Entanglement fidelity F = |Tr(U_out† U_target)|² / d²
        Works for any batch size B and dimension d.
        """
        d = U_out.shape[-1]

        # Trace of U_out† U_target – no data movement thanks to index relabelling.
        inner = torch.einsum("bji,bij->b", U_out.conj(), U_target)  # shape [B]

        return (inner.conj() * inner / d ** 2).real  # shape [B] or scalar

    ###############################################################################
    # Positional Embedding
    ###############################################################################

    @staticmethod
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


class Pipeline(nn.Module):
    """
    Pipeline for the UniversalQOCTransformer model.
    This class is a wrapper around the UniversalQOCTransformer to provide a clean interface.
    """

    def __init__(
            self, 
            model: UniversalQOCTransformer,
            weight_path: str = None, 
            device: Optional[torch.device] = None
        ) -> None:
        super().__init__()
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        self.model.eval()  # Ensure the model is in evaluation mode
        return self.model(rotation_vector).detach()

    def forward_with_unitary(self, unitary: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with a batch of target unitaries.
        unitary: (B, 2, 2) complex tensor
        Returns: (B, ...) output from the model
        """
        self.model.eval()
        # Extract rotation vector for each unitary in the batch
        n_x = torch.real(unitary[:, 0, 1])
        n_y = torch.imag(unitary[:, 0, 1])
        n_z = torch.real(unitary[:, 1, 1])
        theta = torch.acos(torch.real(unitary[:, 0, 0]).clamp(-1.0, 1.0))
        rotation_vector = torch.stack([n_x, n_y, n_z, theta], dim=1)  # (B, 4)
        return self.forward_with_rotation_vector(rotation_vector).detach()
