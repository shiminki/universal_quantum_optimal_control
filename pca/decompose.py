"""SVD/PCA analysis of transformer pulse data matrices.

Finds the effective dimension of the control manifold:
    phi(t; theta, alpha) lies in a low-dimensional affine subspace
    of the full L-dimensional (or 3L-dimensional) pulse space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    """Container for one PCA decomposition."""

    label: str              # "phi", "tau", or "full"
    theta_grid: np.ndarray  # (N_theta,)
    alpha_grid: np.ndarray  # (N_alpha,)
    X_mean: np.ndarray      # (D,)  column mean of the data matrix
    S: np.ndarray           # (K,)  singular values (top K stored)
    Vt: np.ndarray          # (K, D) right singular vectors (PCA basis)
    amplitudes: np.ndarray  # (N_theta, N_alpha, K)  projection coefficients A_j(theta, alpha)
    evr: np.ndarray         # (K,)  explained variance ratios (from all singular values)
    m: int                  # effective dimension at the energy_threshold


def run_pca(
    X: np.ndarray,
    theta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    label: str,
    energy_threshold: float = 0.999,
    max_components: int = 50,
) -> PCAResult:
    """Run centered SVD on data matrix X and return PCA decomposition.

    The effective dimension m is the smallest integer such that
    sum(evr[:m]) >= energy_threshold.

    Args:
        X:                (N, D) data matrix, N = N_theta * N_alpha
        theta_grid:       (N_theta,) polar angles
        alpha_grid:       (N_alpha,) rotation angles
        label:            name of this decomposition ("phi", "tau", "full")
        energy_threshold: cumulative variance threshold (default 99.9%)
        max_components:   maximum number of PCs to store

    Returns:
        PCAResult with full decomposition info
    """
    N_theta = len(theta_grid)
    N_alpha = len(alpha_grid)
    assert X.shape[0] == N_theta * N_alpha, (
        f"X must have {N_theta * N_alpha} rows, got {X.shape[0]}"
    )

    # Center
    X_mean = X.mean(axis=0)
    X_c = X - X_mean

    # Economy SVD: returns min(N, D) singular values
    # For (2000, 1200): computes 1200 singular values
    print(f"  Running SVD on {X_c.shape} matrix (label={label})...")
    _, S_all, Vt_all = np.linalg.svd(X_c, full_matrices=False)

    # Explained variance ratio (use all singular values for accurate total)
    var_all = S_all ** 2
    total_var = var_all.sum()
    evr_all = var_all / (total_var + 1e-30)
    cumulative = np.cumsum(evr_all)

    # Effective dimension
    m_idx = np.searchsorted(cumulative, energy_threshold)
    m = int(min(m_idx + 1, max_components, len(S_all)))

    # Truncate to max_components for storage
    K = min(max_components, len(S_all))
    S = S_all[:K]
    Vt = Vt_all[:K]
    evr = evr_all[:K]

    # Amplitude maps: A = X_c @ Vt.T  shape (N, K)
    amplitudes_flat = X_c @ Vt.T  # (N, K)
    amplitudes = amplitudes_flat.reshape(N_theta, N_alpha, K)

    return PCAResult(
        label=label,
        theta_grid=theta_grid,
        alpha_grid=alpha_grid,
        X_mean=X_mean,
        S=S,
        Vt=Vt,
        amplitudes=amplitudes,
        evr=evr,
        m=m,
    )
