from __future__ import annotations

import math
from typing import Callable, Dict, List, Iterable

import torch

from model_encoder import CompositePulseTransformerEncoder
from trainer import CompositePulseTrainer

###############################################################################
# Pauli matrices and helpers – cached per device
###############################################################################

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)

# Simple †‑immortal cache keyed by torch.device.
_PAULI_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_paulis(device: torch.device) -> torch.Tensor:
    """Return a stack ``(4, 2, 2)`` of *(I, σₓ, σ_y, σ_z)* on *device*.

    The tensors are created on their first use on each device and then reused
    to avoid needless kernel launches and host‑to‑device traffic.
    """
    if device not in _PAULI_CACHE:
        _PAULI_CACHE[device] = torch.stack(
            [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU], dim=0
        ).to(device)
    return _PAULI_CACHE[device]


def batched_unitary_generator_two_qubits(
    pulses: torch.Tensor,
    errors: torch.Tensor,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 4)``, where each pulse is
        ``[Δ, Ω, φ, t]`` (detuning, Rabi amplitude, phase, duration).
    errors : torch.Tensor
        Shape ``(3, B,)``, where [delta1, delta2, epsilon3]

    Returns
    -------
    torch.Tensor
        Shape ``(B, 2, 2)`` complex64/128 – the composite unitary ``U_L ⋯ U_1``.
    """

    if pulses.ndim != 3 or pulses.shape[-1] != 4:
        raise ValueError("'pulses' must have shape (B, L, 4)")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    # Unpack and reshape to broadcast with Pauli matrices.
    Delta, Omega, phi, tau = pulses.unbind(dim=-1)  # each (B, L)

    # (4, 2, 2) on correct device
    pauli = _get_paulis(device).type(dtype)

    # Projectors |0><0| and |1><1|
    proj_0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=pulses.device)
    proj_1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=pulses.device)

    # ORE and PLE
    delta1 = errors[0]
    delta2 = errors[1]
    epsilon = errors[2]


    H_base = (
        torch.kron(proj_0, Delta[..., None, None] * pauli[3]) # |0><0| x Delta Z
        + torch.kron(
            proj_1,
            Omega[..., None, None]
            * (
                torch.cos(phi)[..., None, None] * pauli[1]
                + torch.sin(phi)[..., None, None] * pauli[2]
            )
        ) # |1><1| x Omega exp(i phi) (X, Y)
    )
    
    H = (
        H_base
        + delta1[..., None, None, None] * torch.kron(pauli[3], pauli[0])
        # delta1 * Z x I
        + delta2[..., None, None, None] * torch.kron(pauli[0], pauli[3])
        # delta2 * I x Z
    ) * (1 + epsilon[..., None, None, None]) # PLE

    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 4, 4)

    U_out = torch.eye(4, dtype=dtype, device=device).expand(B, 4, 4)
    # for k in range(L - 1, -1, -1):  # reverse order
    for k in range(L):
        U_out = U[:, k] @ U_out

    return U_out


def get_errors(batch_size:int , delta1_std:float, delta2_std: float, epsilon_std: float):
    errors = [
        torch.randn(batch_size) * error_std
        for error_std in (delta1_std, delta2_std, epsilon_std)
    ]
    return torch.stack(errors)


def fidelity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Entanglement fidelity F = (|Tr(U_out^† U_target)|² + d) / d(d + 1)."""
    # trace over last two dims, keep batch
    # Batched conjugate transpose and matrix multiplication
    U_out_dagger = U_out.conj().transpose(-1, -2)  # [batch, 2, 2]
    product = U_out_dagger @ U_target  # [batch, 2, 2]

    # print(product, product.shape)

    # Batched trace calculation
    trace = torch.einsum('bii->b', product)  # [batch]
    trace_squared = torch.abs(trace) ** 2

    d = 2 ** num_qubits

    return (trace_squared + d) / (d * (d + 1))

def negative_log_loss(U_out, U_target, fidelity_fn, num_qubits):
    return -torch.log(torch.mean(fidelity_fn(U_out, U_target, num_qubits)))



def _rotation_unitary(axis, theta) -> torch.Tensor:
    """Generate a Haar‑random SU(2) element via axis‑angle rotation."""
    # normalize
    if type(axis) is tuple:
        axis = torch.tensor(axis, dtype=torch.float64)
    axis /= axis.norm()
    n_x, n_y, n_z = axis
    H = 0.5 * theta * (n_x * _SIGMA_X_CPU + n_y * _SIGMA_Y_CPU + n_z * _SIGMA_Z_CPU)
    return torch.matrix_exp(-1j * H)



def build_two_qubit_dataset() -> List[torch.Tensor]:
    axis_angles = [
        ((1, 0, 0), torch.pi),   # X(pi)
        ((1, 0, 0), torch.pi/2), # X(pi/2)
        ((1, 0, 1), torch.pi),   # Hadamard
        ((0, 0, 1), torch.pi/4)  # T-gate = Z(pi/4)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gates = [
        torch.kron(_rotation_unitary(axis, theta), _I2_CPU)
        for (axis, theta) in axis_angles
    ]

    gates = [
        torch.kron(_I2_CPU, _rotation_unitary(axis, theta))
        for (axis, theta) in axis_angles
    ]

    # Projectors |0><0| and |1><1|
    proj_0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat)
    proj_1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat)

    gates.append(
        torch.kron(proj_0, _I2_CPU) + torch.kron(proj_1, _SIGMA_X_CPU)
    )
    gates.append(
        torch.kron(_I2_CPU, proj_0) + torch.kron(_SIGMA_X_CPU, proj_1)
    )

    return torch.stack(gates).to(device)


def main():
    pulse_space = {
        "Delta": (0, 5.0),           # detuning [rad s-1]
        "Omega": (0, 1.0),           # Rabi rate (normalised)
        "phi": (-math.pi, math.pi),     # phase [rad]
        "tau": (0.0, 0.3),                # duration [arb. units]
    }

    model_params = {
        "num_qubits" : 2, "pulse_space" : pulse_space, "max_pulses" : 200,
        "d_model" : 256, "n_layers" : 4, "n_heads" : 16, "dropout" : 0.1
    }

    model = CompositePulseTransformerEncoder(**model_params)

    trainer_params = {
        "model" : model, "unitary_generator" : batched_unitary_generator_two_qubits,
        "error_sampler": get_errors,
        "fidelity_fn": fidelity,
        "loss_fn": negative_log_loss,
        # "loss_fn": infidelity_loss,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    trainer = CompositePulseTrainer(**trainer_params)
    train_set = build_two_qubit_dataset()

    for U in train_set:
        print(U)

    error_params_list = [
        {"delta1_std" : delta_std, "delta2_std": delta_std, "epsilon_std": 0.05} 
        for delta_std in torch.arange(0.1, 3.05, 0.1)
    ]

    trainer.train(train_set, error_params_list=error_params_list, epochs=3000, save_path="weights/Two_Qubit_Weights/")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
