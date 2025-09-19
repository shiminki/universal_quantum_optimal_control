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

from model.universal_model import UniversalQOCTransformer
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


_ZZ_CACHE = {}
def _zz(device, dtype):
    key = (device, dtype)
    if key not in _ZZ_CACHE:
        pauli = _get_paulis(device).type(dtype)
        _ZZ_CACHE[key] = torch.kron(pauli[3], pauli[3]).to(device).type(dtype).contiguous()
    return _ZZ_CACHE[key]


###############################################################################
# Batched propagator for a composite‑pulse sequence
###############################################################################


def batched_unitary_generator(
    pulses: torch.Tensor,
    error: torch.Tensor,
    J: float = 0.1,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 5)``, where each pulse is
        ``[Ω_sys, φ_sys, Ω_anc, φ_anc, t]`` (detuning, Rabi amplitude, phase, duration).
    error : torch.Tensor
        Shape ``(4, B,)`` static off‑resonant detuning of system and ancila, and pulse length error for each
        batch element.  If you fuse Monte‑Carlo repeats into the batch, just
        expand ``delta`` accordingly.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 2, 2)`` complex64/128 – the composite unitary ``U_L ⋯ U_1``.
    """

    if pulses.ndim != 3 or pulses.shape[-1] != 5:
        raise ValueError(f"'pulses' must have shape (B, L, 5). Input pusle has shape {pulses.shape}")
    
    if error.ndim != 2 or error.shape[0] != 4:
        raise ValueError("'error' must have shape (4, B)")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    # Unpack and reshape to broadcast with Pauli matrices.
    phi_sys, omega_sys, phi_anc, omega_anc, tau = pulses.unbind(dim=-1)  # each (B, L)

    # (4, 2, 2) on correct device
    pauli = _get_paulis(device).type(dtype)

    # ORE and PLE
    delta_sys = error[0]
    delta_anc = error[1]
    epsilon = error[2]
    coupling_error = error[3]


    def build_single_hamiltonian(omega, phi, delta, epsilon):
        H_base = omega[..., None, None] * (
            torch.cos(phi)[..., None, None] * pauli[1]
            + torch.sin(phi)[..., None, None] * pauli[2]
        )
        H = H_base + delta[..., None, None, None] * pauli[3]
        H = 0.5 * H * (1 + epsilon[..., None, None, None])

        return H

    # Build base Hamiltonian H₀ for every pulse in parallel.
    H_sys = omega_sys[..., None, None] * (
        torch.cos(phi_sys)[..., None, None] * pauli[1]
        + torch.sin(phi_sys)[..., None, None] * pauli[2]
    )
    
    H_sys = build_single_hamiltonian(omega_sys, phi_sys, delta_sys, epsilon)
    H_anc = build_single_hamiltonian(omega_anc, phi_anc, delta_anc, epsilon)

    H_int = 0.5 * (1 + coupling_error[..., None, None, None]) * (J * _zz(device, dtype))  # (4, 4)
    H_int = H_int.to(device).type(dtype)  # (1,1,4,4)

    H = torch.kron(H_sys, _I2_CPU.to(device)) + torch.kron(_I2_CPU.to(device), H_anc) + H_int  # (B, L, 4, 4)

    # U_k = exp(-i H_k t_k)
    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 4, 4)

    # U: (B, L, 4, 4)   want: U[:, L-1] @ ... @ U[:, 1] @ U[:, 0]
    U_out = torch.eye(4, dtype=dtype, device=device).expand(B, 4, 4)

    for k in range(L - 1, -1, -1):
        U_out = U[:, k] @ U_out

    return U_out




###############################################################################
# Off‑resonant‑error (ORE) distribution helper
###############################################################################

def get_ore_error_distribution(batch_size:int, delta_std: float = 1.0) -> torch.tensor:
    return torch.randn(batch_size) * delta_std


def get_error_distribution(batch_size:int, delta_std: float = 1.0, epsilon_std: float=0.05, coupling_std: float=0.1) -> torch.tensor:
    ore_sys_error = torch.randn(batch_size) * delta_std
    ore_anc_error = torch.randn(batch_size) * delta_std
    ple_error = torch.randn(batch_size) * epsilon_std
    coupling_error = torch.randn(batch_size) * coupling_std
    return torch.stack(
        [ore_sys_error, ore_anc_error, ple_error, coupling_error]
    )

###############################################################################
# Loss and fidelity functions
###############################################################################


def fidelity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int=1) -> torch.Tensor:
    """
    Returns fidelity such that U_out is effectively a product state U_target (x) SU(2)

    Parameters
    ----------
    U_out : torch.Tensor
        Shape (B, 4, 4) unitary generated by the model. This unitary acts on both system and ancilla qubit
    U_target : torch.Tensor
        Shape (B, 2, 2) unitary, which is the desired unitary gate on the system qubit
    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` fidelity of each gate

    -----------
    Computation
    -----------
    We can define fidelity to be
    F := max_{W \\in SU(2)} 1/16 * |Tr[(U_target^\\dagger (x) W^\\dagger) * U_out]|^2

    Let U_eff = (U_target^\\dagger (x) I) * U_out] and X = Tr_1[U_eff]
    SVF of X is X = U diag(s1, s2) V^\\dagger

    Notice that the trace is maximized when W = U V^\\dagger. 

    Hence F = 1/16 * (s1 + s2)^2, and the average fidelity is

    F_avg = (4 * F + 1)/5
    """
    assert U_out.shape[1] == U_out.shape[2] == 4, f"U_out is not a 4x4 unitary. Shape: {U_out.shape}"
    assert U_target.shape[1] == U_target.shape[2] == 2, f"U_target is not a 2x2 unitary. Shape: {U_target.shape}"
    assert U_out.shape[0] == U_target.shape[0], f"U_out and U_target not matching in length: {U_out.shape[0]} != {U_target.shape[0]}"

    device = U_out.device
    dtype = U_out.dtype
    B = U_out.shape[0]
    I2 = torch.eye(2, dtype=dtype, device=device).expand(B, 2, 2)

    U_t_dag = U_target.conj().transpose(-2, -1)

    # Extract 2x2 blocks of U_out
    A = U_out[:, 0:2, 0:2]  # (B,2,2)
    B = U_out[:, 0:2, 2:4]
    C = U_out[:, 2:4, 0:2]
    D = U_out[:, 2:4, 2:4]

    # Entries of U_t_dag, shaped for broadcast with 2x2 blocks
    a = U_t_dag[:, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    b = U_t_dag[:, 0, 1].unsqueeze(-1).unsqueeze(-1)
    c = U_t_dag[:, 1, 0].unsqueeze(-1).unsqueeze(-1)
    d = U_t_dag[:, 1, 1].unsqueeze(-1).unsqueeze(-1)

    # U_t_dag (x) I is [[aI, bI], [cI, dI]]

    # X = block00 + block11 of (U_t_dag ⊗ I) U_out, derived directly:
    # block00 = a*A + b*C, block11 = c*B + d*D
    X = a * A + b * C + c * B + d * D  # (B,2,2)

    # Singular values of each X
    S = torch.linalg.svdvals(X)  # (B, 2)

    # Nuclear norm = sum of singular values
    nuc_norm = S.sum(dim=-1)  # (B,)

    F = (nuc_norm ** 2) / 16

    return (2 * F.real + 1) / 3


def negative_log_loss(U_out, U_target, fidelity_fn, num_qubits):
    return -torch.log(torch.mean(fidelity_fn(U_out, U_target, num_qubits)))


def infidelity_loss(U_out, U_target, fidelity_fn, num_qubits):
    return 1 - torch.mean(fidelity_fn(U_out, U_target, num_qubits))


def sharp_loss(U_out, U_target, fidelity_fn, num_qubits, tau=0.99, k=100):
    # F: (B,) per-sample fidelity
    F = fidelity_fn(U_out, U_target, num_qubits=num_qubits)
    # per-sample smooth hinge, then mean
    return custom_loss(F, tau, k).mean()

def custom_loss(x, tau=0.99, k=100):
    # x can be a scalar or a tensor; elementwise is fine
    return torch.log1p(torch.exp(-k * (x - tau))) * (1 - x)




###############################################################################
# data
###############################################################################





def unit_vec(phi):
    n_x, n_y = math.cos(phi), math.sin(phi)
    return (n_x, n_y, 0)


def build_SU2_dataset(dataset_size=10000, random=False) -> List[torch.Tensor]:
    """Generate a batch of random SU(2) rotation vectors."""

    if not random:
        B = int(math.sqrt(dataset_size))  # batch size

        theta_list = torch.linspace(0, math.pi, B)  # polar angle
        alpha_list = torch.linspace(0, 2 * math.pi, dataset_size // B)  # azimuthal angle
        theta, alpha = torch.meshgrid(theta_list, alpha_list, indexing='ij')
        theta = theta.flatten()  # (B²,)
        alpha = alpha.flatten()  # (B²,)
        phi = torch.rand(B * (dataset_size // B)) * 2 * math.pi
    else:
        eps = 1e-2
        theta = torch.rand(dataset_size) * math.pi * (1 + eps)
        alpha = torch.rand(dataset_size) * 2 * math.pi * (1 + eps)
        phi = torch.rand(dataset_size) * 2 * math.pi * (1 + eps)

    # Rotation axis (spherical coordinates)
    n_x = torch.sin(theta) * torch.cos(phi)
    n_y = torch.sin(theta) * torch.sin(phi)
    n_z = torch.cos(theta)
    n = torch.stack([n_x, n_y, n_z], dim=1)  # (B, 3)
    n = n / n.norm(dim=1, keepdim=True)

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

    param_names = list(params['pulse_space'].keys())

    print(f"pulse_names: {param_names}")
    print(f"First element: {param_names[0]}")


    return params


###############################################################################
# Training code
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Train composite pulse model")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="weights/single_qubit_control/weights", help="Path to save model weights")
    parser.add_argument("--delta_control", type=float, default=0.5, help="threshold for delta control; generate identity for |delta| < delta_control")
    args = parser.parse_args()


    # Load model parameters from external JSON
    current_directory = os.path.dirname(__file__)
    print(current_directory)
    model_params = load_model_params(f"{current_directory}/model_params.json")
    model = UniversalQOCTransformer(**model_params)

    # load pretrained module

    # model_path = "weights/phase_control_0.02_tau_max/err_{_delta_std_tensor(0.7000),_epsilon_std_0.05}.pt"
    # model.load_state_dict(torch.load(model_path))

    trainer_params = {
        "model" : model, "unitary_generator" : batched_unitary_generator,
        "error_sampler": get_error_distribution,
        "fidelity_fn": fidelity,
        "loss_fn": sharp_loss,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delta_control": args.delta_control
    }

    trainer = UniversalModelTrainer(**trainer_params)


    # train
    train_size = 12000
    eval_size = 3000
    batch_size = 300

    # # debugging
    # train_size = 24
    # eval_size = 6
    # batch_size = 2
    
    train_rotation_vec, train_unitaries = build_SU2_dataset(dataset_size=train_size, random=True)
    eval_rotation_vec, eval_unitaries = build_SU2_dataset(dataset_size=eval_size, random=True)
    
    # 200 fits ~37GB for len 100 model
    # batch_size = 50 # fits ~37GB GPU memory for len 400 model
    
    
    #####################
    ## Training #########
    #####################


    # 5% PLE error'
    error_params_list = [{"delta_std" : delta_std, "epsilon_std": 0.05} for delta_std in torch.arange(0.4, 1.05, 0.3)]
    
    trainer.train(
        train_rotation_vec,
        train_unitaries,
        eval_rotation_vec,
        eval_unitaries,
        error_params_list=error_params_list,
        epochs=args.num_epoch,
        save_path=args.save_path,
        plot=True,
        batch_size=batch_size
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()