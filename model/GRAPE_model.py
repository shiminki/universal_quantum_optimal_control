import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.pulse_length = num_pulses

        self.device = device
        self.num_qubits = 1

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize neural network parameters for pulse optimization
        self.layer = nn.Linear(8, self.pulse_length * self.num_param, bias=False) # 8 -> L * P

    def forward(self, U_target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRAPE model.

        Args:
            U_target: Input unitary tensor of shape (batch_size, d, d).

        Returns:
            Output tensor after applying the GRAPE model.
        """
        # Apply the GRAPE optimization logic here
        B = U_target.shape[0]  # batch size
        pulse_norm = self.layer(_to_real_vector(U_target))  # shape: (B, L * P)
        pulse_norm = pulse_norm.reshape(B, self.pulse_length, self.num_param) # (B, L, P)

        # Normalize the pulse parameters to their respective ranges
        pulses_unit = pulse_norm.sigmoid()
        low = self.param_ranges[:, 0].to(pulses_unit.device)
        high = self.param_ranges[:, 1].to(pulses_unit.device)

        pulses = low + (high - low) * pulses_unit  # shape: (B, L, P)

        return pulses
