import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap   # <-- 1-line opt-in for vectorised maps

from tqdm import tqdm

__all__ = ["GRAPE"]


def _to_real_vector(U: torch.Tensor) -> torch.Tensor:
    """Flatten a complex matrix into a real‑valued vector with alternating real and imag components (…, 2*d*d)."""
    real = U.real.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)
    imag = U.imag.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)

    stacked = torch.stack((real, imag), dim=-1)  # shape: (..., d*d, 2)
    interleaved = stacked.reshape(*U.shape[:-2], -1)  # shape: (..., 2*d*d)

    return interleaved


class GRAPE(nn.Module):
    """
    GRAPE (Gradient Ascent Pulse Engineering) model for quantum control.
    This model is designed to optimize pulse sequences for quantum systems.
    """

    def __init__(
            self, 
            pulse_space: Dict[str, Tuple[float, float]], 
            num_pulses: int,
            device: torch.device = None
        ):
        super(GRAPE, self).__init__()
        self.param_names = list(pulse_space.keys())
        self.param_ranges: torch.Tensor = torch.tensor(
            [pulse_space[k] for k in self.param_names], dtype=torch.float32
        )  # (P, 2)
        
        self.num_param = self.param_ranges.shape[0]

        assert self.num_param == 2, "Only supports 2 parameters (phase and time) for now."

        self.pulse_length = num_pulses

        self.device = device
        self.num_qubits = 1

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize neural network parameters for pulse optimization
        L = self.pulse_length * 3
        self.layer = nn.Sequential(
            nn.Linear(4, L, bias=False),
            nn.ReLU(),
            nn.Linear(L, L, bias=False)
        )
        # self.layer = nn.Linear(1, L, bias=False)

    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRAPE model.

        Args:
            rotation_vector: shape (B, 4) – target rotation axis and angle in the form of (n_x, n_y, n_z, theta).

        Returns:
            Output tensor after applying the GRAPE model.
        """
        # Apply the GRAPE optimization logic here
        B = rotation_vector.shape[0]  # batch size

        pulse_norm = self.layer(rotation_vector)  # shape: (B, L * 3)
        pulse_norm = pulse_norm.reshape(B, self.pulse_length, 3) # (B, L, 3)

        # Normalize the pulse parameters to their respective ranges
        pulse_norm = pulse_norm.sigmoid() # [ux, uy, tau] in (0, 1)
        phi = torch.atan2(pulse_norm[:, :, 1], pulse_norm[:, :, 0])
        tau = pulse_norm[:, :, 2]
        pulses_unit = torch.stack((phi, tau), dim=-1) # shape: (B, L, 2
        low = self.param_ranges[:, 0].to(pulses_unit.device)
        high = self.param_ranges[:, 1].to(pulses_unit.device)

        pulses = low + (high - low) * pulses_unit  # shape: (B, L, P)

        base_pulse = GRAPE.get_base_pulse(rotation_vector)  # shape: (B, L, 2)

        # Reshape base_pulse to (B, self.pulse_length, 2)
        n = self.pulse_length // 9
        m = self.pulse_length - 9 * n
        B = base_pulse.shape[0]

        # Repeat each (phi, theta) n times, dividing theta by n
        expanded = []
        for i in range(9):
            phi = base_pulse[:, i, 0]  # (B,)
            theta = base_pulse[:, i, 1]  # (B,)
            # Create (B, n, 2): repeat phi, theta/n n times
            block = torch.stack([
                phi.repeat_interleave(n).reshape(B, n),
                (theta / n).repeat_interleave(n).reshape(B, n)
            ], dim=-1)  # (B, n, 2)
            expanded.append(block)
        base_pulse_expanded = torch.cat(expanded, dim=1)  # (B, 9*n, 2)

        # Pad with m items of (0, 0) if needed
        if m > 0:
            pad = torch.zeros((B, m, 2), dtype=base_pulse.dtype, device=base_pulse.device)
            base_pulse_expanded = torch.cat([base_pulse_expanded, pad], dim=1)  # (B, self.pulse_length, 2)

        base_pulse = base_pulse_expanded  # (B, self.pulse_length, 2)



        pulses = base_pulse + pulses  # shape: (B, L, 2)

        pulses[:, :, -1] = F.relu(pulses[:, :, -1])

        return pulses
    

    @staticmethod
    def get_base_pulse(rotation_vector: torch.Tensor) -> torch.Tensor:
        """
        Returns the base pulse for a given rotation vector.
        The base pulse is a fixed sequence of pulses designed to achieve a specific rotation.
        
        Args:
            rotation_vector: shape (B, 4) – target rotation axis and angle in the form of (n_x, n_y, n_z, theta).
        
        Returns:
            Base pulse tensor of shape (B, L, 2).
        """
        # For simplicity, we return a fixed base pulse here.
        # In practice, this could be more complex based on the rotation_vector.
        B = rotation_vector.shape[0]
        # Base pulse from YXY + SCORE

        euler_angles = GRAPE.euler_yxy_from_rotation_vector(rotation_vector)  # (B, 3)
        score_sequence = GRAPE.score_sequence_from_yxy(euler_angles) # (B, L, 2, 2)

        return score_sequence
    


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
                GRAPE.get_score_emb_pulse(phi, theta)
                for phi, theta in zip(phis, thetas)
            ]
            return torch.cat(blocks, dim=0)  # (9,2,2)                           # (9,2,2)

        # vmap lifts the single-sample function to operate on the whole batch
        return torch.stack([_single_sequence(angles)
                          for angles in euler_angles])  # (B,9,

    @staticmethod
    def get_score_emb_pulse(phi: float,
                              angle: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            score_tensor  – (3,2,2) composite implementing the SCORE pulse
            target_unitary – (2,2) ideal rotation R_{n=unit_vec(φ)}(θ)
        """
        theta = torch.pi - angle - torch.asin(0.5 * torch.sin(angle / 2))


        pulses = torch.tensor([
            [phi + torch.pi, theta],
            [phi, phi + 2 * theta],
            [phi + torch.pi, theta]
        ])
        
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






class GRAPE_finetune_X_pi_2(nn.Module):
    """
    GRAPE (Gradient Ascent Pulse Engineering) model for quantum control.
    This model is designed to optimize pulse sequences for quantum systems.
    """

    def __init__(
            self, 
            pulse_space: Dict[str, Tuple[float, float]], 
            num_pulses: int,
            device: torch.device = None
        ):
        super(GRAPE_finetune_X_pi_2, self).__init__()
        self.param_names = list(pulse_space.keys())
        self.param_ranges: torch.Tensor = torch.tensor(
            [pulse_space[k] for k in self.param_names], dtype=torch.float32
        )  # (P, 2)
        
        self.num_param = self.param_ranges.shape[0]

        assert self.num_param == 2, "Only supports 2 parameters (phase and time) for now."

        self.pulse_length = num_pulses

        self.device = device
        self.num_qubits = 1

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize neural network parameters for pulse optimization
        L = self.pulse_length * 3
        self.layer = nn.Sequential(
            nn.Linear(4, L, bias=False),
            nn.ReLU(),
            nn.Linear(L, L, bias=False)
        )
        # self.layer = nn.Linear(1, L, bias=False)


        # ---- Base Pulse: SCORE4(X_pi_2) ----
        angles = torch.tensor([1.55280, 1.42267, 1.78586, 2.07559]) * torch.pi
        Angle = torch.pi / 2
        base_pulse = []
        n = num_pulses // 10

        for i, angle in enumerate(angles):
            Angle += (-1)**(len(angles) - i - 1) * 2 * angle

            t = angle / n

            phi = (i % 2) * torch.pi

            for _ in range(n):
                base_pulse.append([phi, t])
    
        for _ in range(n):
            base_pulse.append([0, Angle / n])

        for i, angle in reversed(list(enumerate(angles))):
            t = angle / n

            phi = (i % 2) * torch.pi

            for _ in range(n):
                base_pulse.append([phi, t])
        
        for _ in range(n):
            base_pulse.append([0, 0])

        self.base_pulse = torch.tensor(base_pulse, dtype=torch.float32).to(self.device) # (L, 2)
        


    def forward(self, rotation_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRAPE model.

        Args:
            rotation_vector: shape (B, 4) – target rotation axis and angle in the form of (n_x, n_y, n_z, theta).

        Returns:
            Output tensor after applying the GRAPE model.
        """
        # Apply the GRAPE optimization logic here
        B = rotation_vector.shape[0]  # batch size
        pulse_norm = self.layer(rotation_vector)  # shape: (B, L * 3)
        pulse_norm = pulse_norm.reshape(B, self.pulse_length, 3) # (B, L, 3)

        # Normalize the pulse parameters to their respective ranges
        pulse_norm = pulse_norm.sigmoid() # [ux, uy, tau] in (0, 1)
        phi = torch.atan2(pulse_norm[:, :, 1], pulse_norm[:, :, 0])
        tau = pulse_norm[:, :, 2]
        pulses_unit = torch.stack((phi, tau), dim=-1) # shape: (B, L, 2
        low = self.param_ranges[:, 0].to(pulses_unit.device)
        high = self.param_ranges[:, 1].to(pulses_unit.device)

        pulses = low + (high - low) * pulses_unit  # shape: (B, L, P)
        pulses += self.base_pulse.unsqueeze(0)

        pulses[:, :, -1] = F.relu(pulses[:, :, -1])

        return pulses
