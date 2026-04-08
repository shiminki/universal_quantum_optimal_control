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

import json
import argparse

import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from model.GRAPE_model import GRAPE
from model.universal_model_trainer import UniversalModelTrainer


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


def batched_unitary_generator(
    pulses: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 2)``, where each pulse is
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

    if pulses.ndim != 3 or pulses.shape[-1] != 2:
        raise ValueError("'pulses' must have shape (B, L, 2)")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    # Unpack and reshape to broadcast with Pauli matrices.
    phi, tau = pulses.unbind(dim=-1)  # each (B, L)

    # ORE and PLE
    delta = error[0]
    epsilon = error[1]

    # Closed-form 2×2 propagator (Rodrigues' rotation formula).
    #   H   = 0.5·(1+ε) · (cos φ σ_x + sin φ σ_y + δ σ_z)
    #   v   = 0.5·(1+ε)·τ · (cos φ, sin φ, δ)
    #   α   = |v|
    #   U   = cos α · I − i (sin α / α) (v·σ)
    # Avoids torch.linalg.matrix_exp on tens of thousands of 2×2 matrices,
    # which is the dominant cost of the previous implementation.
    A = 0.5 * (1.0 + epsilon[..., None]) * tau                  # (B, L)
    v_x = A * torch.cos(phi)                                     # (B, L)
    v_y = A * torch.sin(phi)                                     # (B, L)
    v_z = A * delta[..., None]                                   # (B, L)
    alpha = torch.sqrt(v_x * v_x + v_y * v_y + v_z * v_z)        # (B, L)

    cos_a = torch.cos(alpha)
    sinc_a = torch.sinc(alpha / math.pi)  # sin(α)/α, numerically safe at α=0

    u00 = (cos_a - 1j * sinc_a * v_z).to(dtype)
    u01 = (-sinc_a * (1j * v_x + v_y)).to(dtype)
    u10 = (-sinc_a * (1j * v_x - v_y)).to(dtype)
    u11 = (cos_a + 1j * sinc_a * v_z).to(dtype)

    U = torch.stack(
        [torch.stack([u00, u01], dim=-1),
         torch.stack([u10, u11], dim=-1)],
        dim=-2,
    )  # (B, L, 2, 2)

    X = U
    I = torch.eye(2, dtype=dtype, device=device).expand(B, 1, 2, 2)

    while X.size(1) > 1:
        # pad to even length
        if (X.size(1) & 1) == 1:
            X = torch.cat([X, I], dim=1)
        # pairwise multiply preserving left-to-right order:
        # (U1 @ U0), (U3 @ U2), ...
        X = X[:, 1::2] @ X[:, 0::2]

    U_out = X[:, 0]  # (B, 2, 2)


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


def sharp_loss(U_out, U_target, fidelity_fn, num_qubits, tau=0.99, k=100):
    F = torch.mean(fidelity_fn(U_out, U_target, num_qubits))
    return custom_loss(F, tau, k)

def custom_loss(x, tau=0.99, k=100):
    return torch.log(1 + torch.exp(-k * (x - tau))) * (1 - x)



###############################################################################
# data
###############################################################################


def build_SU2_dataset(batch_size=10000, random=False) -> List[torch.Tensor]:
    """Generate a batch of random SU(2) rotation vectors."""

    if not random:
        B = int(math.sqrt(batch_size))  # batch size

        theta_list = torch.linspace(0, math.pi, B)  # polar angle
        alpha_list = torch.linspace(0, 2 * math.pi, B)  # azimuthal angle
        theta, alpha = torch.meshgrid(theta_list, alpha_list, indexing='ij')
        theta = theta.flatten()  # (B²,)
        alpha = alpha.flatten()  # (B²,)
        phi = torch.rand(B ** 2) * 2 * math.pi
    else:
        theta = torch.rand(batch_size) * math.pi
        alpha = torch.rand(batch_size) * 2 * math.pi
        phi = torch.rand(batch_size) * 2 * math.pi

    # Rotation axis (spherical coordinates)
    n_x = torch.sin(theta) * torch.cos(phi)
    n_y = torch.sin(theta) * torch.sin(phi)
    n_z = torch.cos(theta)
    n = torch.stack([n_x, n_y, n_z], dim=1)  # (B, 3)
    
     # Rotation vector for the function: (n_x, n_y, n_z, alpha)
    rotation_vector = torch.cat([n, alpha.unsqueeze(1)], dim=1).to(torch.float)  # (B, 4)
    
    # Input unitaries
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    sigma_n = n[:, 0, None, None] * X + n[:, 1, None, None] * Y + n[:, 2, None, None] * Z  # (B, 2, 2)
    alpha_half = alpha / 2
    U_input = torch.matrix_exp(-1j * sigma_n * alpha_half[:, None, None])  # (B, 2, 2)


    return rotation_vector, U_input


###############################################################################
# Config loading
###############################################################################


def load_model_params(json_path: str) -> dict:
    with open(json_path, "r") as f:
        params = json.load(f)

    # Convert any stringified tuples to tuples (e.g., for pulse_space ranges)
    if "pulse_space" in params:
        for k, v in params["pulse_space"].items():
            params["pulse_space"][k] = tuple(v)

    return params


###############################################################################
# Training code
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Train composite pulse model")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="weights/single_qubit_control/weights", help="Path to save model weights")
    args = parser.parse_args()


    # Load model parameters from external JSON
    model_params = load_model_params("train/GRAPE/model_params.json")

    # CHOOSE MODEL
    # model = GRAPE_finetune_X_pi_2(**model_params)
    model = GRAPE(**model_params)

    # load pretrained module

    # model_path = "weights/phase_control_0.02_tau_max/err_{_delta_std_tensor(0.7000),_epsilon_std_0.05}.pt"
    # model.load_state_dict(torch.load(model_path))

    trainer_params = {
        "model" : model, "unitary_generator" : batched_unitary_generator,
        "error_sampler": get_ore_ple_error_distribution,
        "fidelity_fn": fidelity,
        "loss_fn": infidelity_loss,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = UniversalModelTrainer(**trainer_params)

    # DEBUG
    # _, train_unitaries = build_SU2_dataset(batch_size=128, random=True)
    # _, eval_unitaries = build_SU2_dataset(batch_size=32, random=True)
    # batch_size = 16

    _, train_unitaries = build_SU2_dataset(batch_size=8192, random=True)
    _, eval_unitaries = build_SU2_dataset(batch_size=1024, random=True)
    batch_size = 256

    #####################
    ## Training #########
    #####################


    # 5% PLE error'
    # error_params_list = [{"delta_std" : delta_std, "epsilon_std": 0.05} for delta_std in torch.arange(0.4, 1.05, 0.3)]
    error_params_list = [{"delta_std" : 1.0, "epsilon_std": 0.05}]
    

    train_unitaries_copy = train_unitaries.clone()
    eval_unitaries_copy = eval_unitaries.clone()

    trainer.train(
        train_unitaries, # use naive C^(2x2) rather than rotation vector for GRAPE NN
        train_unitaries_copy, # aliasing error
        eval_unitaries,
        eval_unitaries_copy,
        error_params_list=error_params_list,
        epochs=args.num_epoch,
        save_path=args.save_path,
        plot=True,
        batch_size=batch_size
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    main()