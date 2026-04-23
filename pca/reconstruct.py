"""Pulse reconstruction from PCA decomposition and fidelity validation.

Reconstruction formula:
    x(theta, alpha) = X_mean + sum_{j=0}^{n-1} A_j(theta, alpha) * V_j

where x encodes [cos(phi_i), sin(phi_i), tau_norm_i] and the inverse gives:
    phi_i  = atan2(x[L + i], x[i])
    tau_i  = x[2L + i] * (tau_max - tau_min) + tau_min
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import (
    batched_unitary_generator,
    fidelity as compute_fidelity,
)

# Pauli matrices for target unitary construction (cached)
_SIGMA_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)


# ---------------------------------------------------------------------------
# Target unitary helpers
# ---------------------------------------------------------------------------

def target_unitary_single(theta: float, alpha: float) -> torch.Tensor:
    """Build target unitary U = exp(-i * alpha/2 * n·sigma) for n = (sin(theta), 0, cos(theta)).

    Matches the convention in app.py:
        H = 0.5 * alpha * (n_x * sigma_x + n_z * sigma_z)
        U = matrix_exp(-1j * H)

    Returns:
        (2, 2) complex tensor
    """
    n_x = math.sin(theta)
    n_z = math.cos(theta)
    H = 0.5 * alpha * (n_x * _SIGMA_X + n_z * _SIGMA_Z)
    return torch.matrix_exp(-1j * H)  # (2, 2)


def target_unitary_batch(theta_vals: np.ndarray, alpha_vals: np.ndarray) -> torch.Tensor:
    """Build a batch of target unitaries.

    Returns:
        (N, 2, 2) complex tensor
    """
    return torch.stack([
        target_unitary_single(float(th), float(al))
        for th, al in zip(theta_vals, alpha_vals)
    ])  # (N, 2, 2)


# ---------------------------------------------------------------------------
# Fidelity evaluation
# ---------------------------------------------------------------------------

def eval_zero_error_fidelity(
    pulses: np.ndarray,
    theta_vals: np.ndarray,
    alpha_vals: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate zero-error fidelity for a batch of pulses vs their target unitaries.

    Zero-error = delta=0, epsilon=0 (no off-resonance, no pulse-length error).
    This measures the ideal gate accuracy.

    Args:
        pulses:     (N, L, 2) or (L, 2) array of [phi_i, tau_i]
        theta_vals: (N,) polar angles of rotation axes
        alpha_vals: (N,) rotation angles

    Returns:
        (N,) fidelity values in [0, 1]
    """
    if pulses.ndim == 2:
        pulses = pulses[np.newaxis]
        theta_vals = np.atleast_1d(theta_vals)
        alpha_vals = np.atleast_1d(alpha_vals)

    N = len(pulses)
    all_fids = []

    for i in range(0, N, batch_size):
        batch_p = pulses[i : i + batch_size]
        batch_th = theta_vals[i : i + batch_size]
        batch_al = alpha_vals[i : i + batch_size]
        B = len(batch_p)

        pulses_t = torch.tensor(batch_p, dtype=torch.float32)
        error = torch.zeros(2, B)

        U_out = batched_unitary_generator(pulses_t, error)  # (B, 2, 2)
        U_targets = target_unitary_batch(batch_th, batch_al)  # (B, 2, 2)

        F = compute_fidelity(U_out, U_targets, num_qubits=1)  # (B,)
        all_fids.append(F.detach().cpu().numpy())

    return np.concatenate(all_fids)


# ---------------------------------------------------------------------------
# PulseReconstructor
# ---------------------------------------------------------------------------

class PulseReconstructor:
    """Reconstruct transformer-equivalent pulses from PCA + Fourier amplitude fit.

    Uses the 'full' encoding PCA result (X_full = [cos(phi), sin(phi), tau_norm]):
        x(theta, alpha) = X_mean + sum_{j=0}^{n-1} A_j(theta, alpha) * V_j

    Recovery of physical parameters:
        phi_i  = atan2(x[L + i], x[i])                       (circular inversion)
        tau_i  = x[2L + i] * (tau_max - tau_min) + tau_min   (linear rescaling)
    """

    def __init__(
        self,
        pca_result,       # PCAResult for the "full" encoding
        fit_result,       # FitResult from fit_2d_amplitudes
        L: int = 400,
        tau_min: float = 0.02,
        tau_max: float = 0.3,
    ):
        self.pca = pca_result
        self.fit = fit_result
        self.L = L
        self.tau_min = tau_min
        self.tau_max = tau_max

    def get_pulse(
        self,
        theta: float,
        alpha: float,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct pulse [phi_i, tau_i]_{i=1}^L at a single (theta, alpha) point.

        Args:
            theta:       polar angle of rotation axis
            alpha:       rotation angle
            n_components: PCA components to use (default: m from PCA analysis)

        Returns:
            (L, 2) array of [phi_i, tau_i]
        """
        n = self._resolve_n(n_components)
        A = self.fit.predict(theta, alpha)  # (m,) — Fourier evaluation

        # Reconstruct encoded vector
        x = self.pca.X_mean.copy()
        for j in range(n):
            x += float(A[j]) * self.pca.Vt[j]

        return self._decode(x)

    def get_pulses_batch(
        self,
        theta_vals: np.ndarray,
        alpha_vals: np.ndarray,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """Batch reconstruct for multiple (theta, alpha) points.

        Fully vectorized — avoids Python loop over N.

        Args:
            theta_vals: (N,) polar angles
            alpha_vals: (N,) rotation angles
            n_components: PCA components to use

        Returns:
            (N, L, 2) array of [phi_i, tau_i] pulses
        """
        n = self._resolve_n(n_components)

        # A: (N, n) Fourier-predicted amplitudes
        A = self.fit.predict_batch(theta_vals, alpha_vals)[:, :n]  # (N, n)

        # Reconstruct: x = X_mean + A @ Vt[:n]  shape (N, D)
        x = self.pca.X_mean[np.newaxis, :] + A @ self.pca.Vt[:n]  # (N, D=3L)

        return self._decode_batch(x)

    def eval_fidelity(
        self,
        theta: float,
        alpha: float,
        n_components: Optional[int] = None,
    ) -> float:
        """Zero-error fidelity of the reconstructed pulse vs target unitary."""
        pulse = self.get_pulse(theta, alpha, n_components)  # (L, 2)
        F = eval_zero_error_fidelity(
            pulse[np.newaxis], np.array([theta]), np.array([alpha])
        )
        return float(F[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_n(self, n_components: Optional[int]) -> int:
        if n_components is None:
            n_components = self.pca.m
        return int(min(n_components, self.fit.m, self.pca.Vt.shape[0]))

    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Decode a single encoded vector x (D,) to pulse (L, 2)."""
        L = self.L
        cos_phi = x[:L]
        sin_phi = x[L : 2 * L]
        tau_norm = x[2 * L : 3 * L]

        phi = np.arctan2(sin_phi, cos_phi)
        tau = tau_norm * (self.tau_max - self.tau_min) + self.tau_min
        tau = np.clip(tau, self.tau_min, self.tau_max)

        return np.stack([phi, tau], axis=1)  # (L, 2)

    def _decode_batch(self, X: np.ndarray) -> np.ndarray:
        """Decode a batch of encoded vectors X (N, D) to pulses (N, L, 2)."""
        L = self.L
        cos_phi = X[:, :L]
        sin_phi = X[:, L : 2 * L]
        tau_norm = X[:, 2 * L : 3 * L]

        phi = np.arctan2(sin_phi, cos_phi)  # (N, L)
        tau = tau_norm * (self.tau_max - self.tau_min) + self.tau_min
        tau = np.clip(tau, self.tau_min, self.tau_max)  # (N, L)

        return np.stack([phi, tau], axis=2)  # (N, L, 2)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_reconstruction(
    pipeline,
    reconstructor: PulseReconstructor,
    theta_test: np.ndarray,
    alpha_test: np.ndarray,
    batch_size: int = 128,
) -> dict:
    """Validate reconstruction quality on a held-out test grid.

    Args:
        pipeline:       Pipeline for model inference
        reconstructor:  PulseReconstructor
        theta_test:     (N_test,) test polar angles
        alpha_test:     (N_test,) test rotation angles

    Returns:
        dict with keys:
            theta, alpha: test coordinates
            fidelity_model:  (N_test,) model zero-error fidelity
            fidelity_recon:  (N_test,) reconstructor zero-error fidelity
            l2_phi:          (N_test,) mean L2 error in phi space (rad)
    """
    N_test = len(theta_test)
    n_x = np.sin(theta_test).astype(np.float32)
    n_z = np.cos(theta_test).astype(np.float32)
    rotation_vecs = np.stack(
        [n_x, np.zeros_like(n_x), n_z, alpha_test.astype(np.float32)], axis=1
    )  # (N_test, 4)

    # Get model pulses
    all_pulses = []
    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            batch = torch.tensor(rotation_vecs[i : i + batch_size], dtype=torch.float32)
            p = pipeline(batch)
            all_pulses.append(p.cpu().numpy())
    model_pulses = np.concatenate(all_pulses, axis=0)  # (N_test, L, 2)

    # Model fidelity
    model_fid = eval_zero_error_fidelity(model_pulses, theta_test, alpha_test)

    # Reconstructed pulses (batched)
    recon_pulses = reconstructor.get_pulses_batch(theta_test, alpha_test)  # (N_test, L, 2)

    # Reconstruction fidelity
    recon_fid = eval_zero_error_fidelity(recon_pulses, theta_test, alpha_test)

    # L2 error in phi space
    phi_model = model_pulses[:, :, 0]  # (N_test, L)
    phi_recon = recon_pulses[:, :, 0]  # (N_test, L)
    # Use circular difference to handle wrap-around at ±pi
    dphi = np.angle(np.exp(1j * (phi_model - phi_recon)))
    l2_phi = np.sqrt(np.mean(dphi ** 2, axis=1))  # (N_test,)

    return {
        "theta": theta_test,
        "alpha": alpha_test,
        "fidelity_model": model_fid,
        "fidelity_recon": recon_fid,
        "l2_phi": l2_phi,
    }
