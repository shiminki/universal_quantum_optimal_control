"""Grid sampling and batched model inference for PCA analysis.

Samples the (theta, alpha) parameter space and collects transformer
outputs [phi_i, tau_i]_{i=1}^{L} for each gate target.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from model.universal_model import UniversalQOCTransformer, Pipeline
from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import load_model_params


def load_pipeline(model_size: int = 400) -> Tuple[Pipeline, float, float]:
    """Load the pretrained transformer pipeline.

    Returns:
        pipeline: Pipeline object ready for inference
        tau_min: minimum tau value from model params
        tau_max: maximum tau value from model params
    """
    params = load_model_params(f"demo_universal/params/length_{model_size}.json")
    tau_min, tau_max = params["pulse_space"]["tau"]
    pipeline = Pipeline(
        UniversalQOCTransformer(**params),
        weight_path=f"demo_universal/weight/length_{model_size}.pt",
    )
    return pipeline, float(tau_min), float(tau_max)


def build_sampling_grid(
    N_theta: int = 40,
    N_alpha: int = 50,
    theta_eps: float = 0.02,
    alpha_eps: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a uniform (theta, alpha) grid on the x-z plane (n_y = 0).

    The rotation axis is n = (sin(theta), 0, cos(theta)), so the model
    input azimuthal angle phi_az = atan2(0, sin(theta)) = 0 for all
    theta in (0, pi), meaning no global phase shift is added.

    Args:
        N_theta: number of polar angle samples
        N_alpha: number of rotation angle samples
        theta_eps: margin to avoid gimbal-lock at theta=0, pi
        alpha_eps: margin to avoid identity degenerate point at alpha=0, 2pi

    Returns:
        theta_grid:    (N_theta,) polar angles in [theta_eps, pi - theta_eps]
        alpha_grid:    (N_alpha,) rotation angles in [alpha_eps, 2*pi - alpha_eps]
        rotation_vecs: (N_theta * N_alpha, 4) array of [n_x, n_y, n_z, alpha]
    """
    theta_grid = np.linspace(theta_eps, math.pi - theta_eps, N_theta)
    alpha_grid = np.linspace(alpha_eps, 2.0 * math.pi - alpha_eps, N_alpha)

    TH, AL = np.meshgrid(theta_grid, alpha_grid, indexing="ij")  # (N_theta, N_alpha)
    TH_flat = TH.ravel()
    AL_flat = AL.ravel()

    n_x = np.sin(TH_flat).astype(np.float32)
    n_y = np.zeros_like(n_x)
    n_z = np.cos(TH_flat).astype(np.float32)
    rotation_vecs = np.stack([n_x, n_y, n_z, AL_flat.astype(np.float32)], axis=1)  # (N, 4)

    return theta_grid, alpha_grid, rotation_vecs


def build_data_matrix(
    pipeline: Pipeline,
    rotation_vecs: np.ndarray,
    batch_size: int = 256,
    tau_min: float = 0.02,
    tau_max: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference in batches and build PCA-ready data matrices.

    Encoding choices:
    - phi: cos/sin encoded as [cos(phi_i), sin(phi_i)] to handle periodicity.
      Avoids branch-cut artifacts at phi = ±pi in PCA distance computations.
    - tau: linearly normalized to [0, 1] to equalize scale with phi encoding.

    Args:
        pipeline:      loaded Pipeline object
        rotation_vecs: (N, 4) array of [n_x, n_y, n_z, alpha] inputs
        batch_size:    number of samples per forward pass
        tau_min, tau_max: tau range from model params

    Returns:
        raw_pulses: (N, L, 2)  raw [phi_i, tau_i] from model
        X_phi:     (N, 2*L)   [cos(phi_i), sin(phi_i)] encoding
        X_tau:     (N, L)     tau normalized to [0, 1]
        X_full:    (N, 3*L)   concatenation of X_phi and X_tau
    """
    N = len(rotation_vecs)
    all_pulses = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.tensor(rotation_vecs[i : i + batch_size], dtype=torch.float32)
            pulses = pipeline(batch)  # (B, L, 2)
            all_pulses.append(pulses.cpu().numpy())

    raw_pulses = np.concatenate(all_pulses, axis=0)  # (N, L, 2)
    phi_raw = raw_pulses[:, :, 0]  # (N, L)
    tau_raw = raw_pulses[:, :, 1]  # (N, L)

    # cos/sin encoding for phi
    X_phi = np.concatenate([np.cos(phi_raw), np.sin(phi_raw)], axis=1)  # (N, 2L)

    # normalize tau to [0, 1]
    X_tau = (tau_raw - tau_min) / (tau_max - tau_min)
    X_tau = np.clip(X_tau, 0.0, 1.0)  # (N, L)

    # full concatenation
    X_full = np.concatenate([X_phi, X_tau], axis=1)  # (N, 3L)

    return raw_pulses, X_phi, X_tau, X_full
