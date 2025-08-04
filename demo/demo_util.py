import torch

import sys
import os
import glob
from scipy.linalg import expm
from scipy.optimize import minimize


# Add parent directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))


from train.unitary_single_qubit_gate.unitary_single_qubit_gate import *
from visualize.util import *



_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]


def generate_unitary(pulse, delta, epsilon):
    phi = pulse[1]
    tau = pulse[2] / 2
    H_base = (np.cos(phi) * pauli[1] + np.sin(phi) * pauli[2])
    H = (H_base + delta * pauli[3])
    return torch.linalg.matrix_exp(-1j * H * tau * (1 + epsilon))


# Convert spinor to Bloch vector
def spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    assert psi.shape == (2,), "psi must be a 2D complex vector"
    assert torch.is_complex(psi), "psi must be complex-valued"
    alpha, beta = psi[0], psi[1]
    x = 2 * torch.real(torch.conj(alpha) * beta)
    y = 2 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha)**2 - torch.abs(beta)**2
    return np.array([x.item(), y.item(), z.item()])


def Rx(theta):
    return expm(-1j * _SIGMA_X_CPU * theta / 2)

def Ry(theta):
    return expm(-1j * _SIGMA_Y_CPU * theta / 2)

def decompose_SU2(U_target):
    def loss(params):
        alpha, beta, gamma = params
        U = Rx(alpha) @ Ry(beta) @ Rx(gamma)
        return np.linalg.norm(U - U_target)

    res = minimize(loss, x0=[0.0, 0.0, 0.0])
    return res.x  # returns alpha, beta, gamma


def euler_yxy_from_axis_angle(nx, ny, nz, theta, *, eps=1e-12):
    """
    Return Euler angles (alpha, beta, gamma) for the y‑x‑y sequence so that
        R_y(alpha) R_x(beta) R_y(gamma)
    equals a rotation of angle `theta` about the axis (nx, ny, nz).

    Handles every special case: θ≈0, θ≈π, axis along ±y, axis in x‑z plane.
    """
    # ---- normalise the axis ------------------------------------------------
    n = np.array([nx, ny, nz], dtype=float)
    n /= np.linalg.norm(n)
    nx, ny, nz = n

    # ---- useful trig shorthands -------------------------------------------
    c = np.cos(theta)          # cos θ
    s = np.sin(theta)          # sin θ
    k = 1.0 - c                # = 2 sin²(θ/2)

    # ---- middle angle β ----------------------------------------------------
    cos_beta = np.clip(c + k * ny * ny, -1.0, 1.0)
    beta     = np.arccos(cos_beta)
    sin_beta = np.sin(beta)

    # -----------------------------------------------------------------------
    # 1) Generic case  (sin β not tiny)  →  use the closed‑form directly
    # -----------------------------------------------------------------------
    if abs(sin_beta) > eps:
        alpha = np.arctan2(k * nx * ny - s * nz,
                           k * ny * nz + s * nx)
        gamma = np.arctan2(k * nx * ny + s * nz,
                           s * nx       - k * ny * nz)
        return alpha, beta, gamma

    # -----------------------------------------------------------------------
    # 2) Singular cases  (sin β ≈ 0)  →  β is 0 or π
    # -----------------------------------------------------------------------
    if abs(ny) > 1 - eps:          # axis almost parallel to ±y
        # Rotation sits entirely on the y‑axis
        alpha = 0.0
        beta  = 0.0
        gamma = +theta if ny > 0 else -theta
        return alpha, beta, gamma

    # Remaining possibility:  ny ≈ 0  and  β ≈ π  (axis in x‑z plane,
    # half‑turn).  Keep α = 0 and solve γ from the x–z block.
    alpha = 0.0
    beta  = np.pi
    gamma = np.arctan2(k * nx * nz,          #  2 nx nz   when θ = π
                       c + k * nx * nx)      # -1+2 nx²   when θ = π
    return alpha, beta, gamma


def get_pulse(axis, theta, model_option):
    if model_option == "100 length":
        model_path = "demo/weight/length_100.pt"
        model_params = load_model_params("demo/params/length_100.json")
    else:
        model_path = "demo/weight/length_400.pt"
        model_params = load_model_params("demo/params/length_400.json")

    theta = math.pi * np.round(theta/math.pi, 3)

    model = CompositePulseTransformerEncoder(**model_params)

    phi = np.arctan2(axis[1], axis[0])

    # load pretrained module

    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, U_target = get_score_emb_unitary(phi, theta)
    SCORE_tensor_base, _ = get_score_emb_unitary(0, theta)

    pulse = model(SCORE_tensor_base.unsqueeze(0)).squeeze(0).detach()

    pulse[:, 0] += phi
    return pulse