r"""
Single‑qubit *Off‑Resonant‑Error* (ORE) control utilities
========================================================

This file provides a **batched, GPU‑friendly** implementation of the unitary
propagator for a composite‑pulse sequence in the presence of a *static* ORE
(`delta`).  It is designed to plug straight into the new
``CompositePulseTrainer`` API introduced in *composite_pulse_model.py* – simply
pass :pyfunc:`batched_unitary_generator` as the ``unitary_generator`` argument
and a distribution created with :pyfunc:`get_ore_error_distribution`` as the
``error_sampler``.

The module follows three design rules to keep the GPU happy:

1.  **No per‑pulse Python loops on the hot‑path.**  The entire Hamiltonian batch
    is built in parallel; the only remaining loop is the left‑to‑right product
    over the (short) pulse length *L*.
2.  **Constants cached per‑device.**  Pauli operators are moved to each new
    device once and reused thereafter – no thousands of host→device copies per
    epoch.
3.  **Monte‑Carlo samples fused into the batch dimension.**  The caller stacks
    the MC draws before invoking :pyfunc:`batched_unitary_generator`, so the
    function never sees the extra dimension; one kernel launch does all the
    work.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

from model_decoder import CompositePulseTransformerDecoder
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

###############################################################################
# Batched propagator for a composite‑pulse sequence
###############################################################################

def batched_unitary_generator_ore(
    pulses: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 4)``, where each pulse is
        ``[Δ, Ω, φ, t]`` (detuning, Rabi amplitude, phase, duration).
    delta : torch.Tensor
        Shape ``(B,)`` *or* ``(B, 1)`` – static off‑resonant detuning for each
        batch element.  If you fuse Monte‑Carlo repeats into the batch, just
        expand ``delta`` accordingly.

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

    # Build base Hamiltonian H₀ for every pulse in parallel.
    H_base = (
        Delta[..., None, None] * pauli[3]
        + Omega[..., None, None]
        * (
            torch.cos(phi)[..., None, None] * pauli[1]
            + torch.sin(phi)[..., None, None] * pauli[2]
        )
    )
    H = H_base + delta[..., None, None, None] * pauli[3]

    # U_k = exp(-i H_k t_k)
    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 2, 2)

    # Left‑to‑right ordered product: U_L ⋯ U_1.
    # (We keep the small Python loop – L ≤ 16 – because matmul reduction is
    # not yet natively supported in TorchScript; the overhead is negligible.)
    U_out = torch.eye(2, dtype=dtype, device=device).expand(B, 2, 2)
    # for k in range(L - 1, -1, -1):  # reverse order
    for k in range(L):
        U_out = U[:, k] @ U_out

    return U_out


def batched_unitary_generator(
    pulses: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 4)``, where each pulse is
        ``[Δ, Ω, φ, t]`` (detuning, Rabi amplitude, phase, duration).
    error : torch.Tensor
        Shape ``(2, B,)`` static off‑resonant detuning and pulse length error for each
        batch element.  If you fuse Monte‑Carlo repeats into the batch, just
        expand ``delta`` accordingly.

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

    # ORE and PLE
    delta = error[0]
    epsilon = error[1]

    # Build base Hamiltonian H₀ for every pulse in parallel.
    H_base = (
        Delta[..., None, None] * pauli[3]
        + Omega[..., None, None]
        * (
            torch.cos(phi)[..., None, None] * pauli[1]
            + torch.sin(phi)[..., None, None] * pauli[2]
        )
    )
    H = H_base + delta[..., None, None, None] * pauli[3]

    if epsilon is not None:
        H = H * (1 + epsilon[..., None, None, None])

    # U_k = exp(-i H_k t_k)
    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 2, 2)

    # Left‑to‑right ordered product: U_L ⋯ U_1.
    # (We keep the small Python loop – L ≤ 16 – because matmul reduction is
    # not yet natively supported in TorchScript; the overhead is negligible.)
    U_out = torch.eye(2, dtype=dtype, device=device).expand(B, 2, 2)
    # for k in range(L - 1, -1, -1):  # reverse order
    for k in range(L):
        U_out = U[:, k] @ U_out

    return U_out


###############################################################################
# Off‑resonant‑error (ORE) distribution helper
###############################################################################

def get_ore_error_distribution(batch_size:int, delta_std: float = 1.0) -> torch.tensor:
    return torch.randn(batch_size) * delta_std


def get_ore_ple_error_distribution(batch_size:int, delta_std: float = 1.0, epsilon_std: float=0.05) -> torch.tensor:
    ore_error = torch.randn(batch_size) * delta_std
    ple_error = torch.randn(batch_size) * epsilon_std
    return torch.stack([ore_error, ple_error])

###############################################################################
# Loss and fidelity functions
###############################################################################


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


def infidelity_loss(U_out, U_target, fidelity_fn, num_qubits):
    return 1 - torch.mean(fidelity_fn(U_out, U_target, num_qubits))



###############################################################################
# data
###############################################################################



def _rotation_unitary(axis, theta) -> torch.Tensor:
    """Generate a Haar‑random SU(2) element via axis‑angle rotation."""
    # normalize
    if type(axis) is tuple:
        axis = torch.tensor(axis, dtype=torch.float64)
    axis /= axis.norm()
    n_x, n_y, n_z = axis
    H = 0.5 * theta * (n_x * _SIGMA_X_CPU + n_y * _SIGMA_Y_CPU + n_z * _SIGMA_Z_CPU)
    return torch.matrix_exp(-1j * H)


def build_dataset() -> List[torch.Tensor]:
    axis_angles = [
        ((1, 0, 0), torch.pi),   # X(pi)
        ((1, 0, 0), torch.pi/2), # X(pi/2)
        ((1, 0, 1), torch.pi),   # Hadamard
        ((0, 0, 1), torch.pi/4)  # T-gate = Z(pi/4)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.stack([
        _rotation_unitary(axis, theta) 
        for (axis, theta) in axis_angles
    ]).to(device)


###############################################################################
# Training code
###############################################################################


def main():
    pulse_space = {
        "Delta": (0, 5.0),           # detuning [rad s-1]
        "Omega": (0, 1.0),           # Rabi rate (normalised)
        "phi": (-math.pi, math.pi),     # phase [rad]
        "tau": (0.0, 0.3),                # duration [arb. units]
    }

    model_params = {
        "num_qubits" : 1, "pulse_space" : pulse_space, "max_pulses" : 200,
        "d_model" : 256, "n_layers" : 4, "n_heads" : 16, "dropout" : 0.1
    }

    model = CompositePulseTransformerEncoder(**model_params)

    trainer_params = {
        "model" : model, "unitary_generator" : batched_unitary_generator,
        "error_sampler": get_ore_ple_error_distribution,
        "fidelity_fn": fidelity,
        "loss_fn": negative_log_loss,
        # "loss_fn": infidelity_loss,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    trainer = CompositePulseTrainer(**trainer_params)
    train_set = build_dataset()

    #####################
    ## Training #########
    #####################


    # 5% PLE error
    error_params_list = [{"delta_std" : delta_std, "epsilon_std": 0.05} for delta_std in torch.arange(0.1, 3.05, 0.1)]

    trainer.train(train_set, error_params_list=error_params_list, epochs=3000, save_path="weights/100_pulses_ORE+PLE/")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
