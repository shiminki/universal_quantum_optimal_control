"""2D Fourier amplitude fitting for PCA coefficient functions A_j(theta, alpha).

Fits a separable 2D Fourier series to each amplitude function:

    A_j(theta, alpha) = c_0
        + sum_{p=1}^{P} c_p  * cos(p * theta)
        + sum_{q=1}^{Q} [a_q * cos(q * alpha) + b_q * sin(q * alpha)]
        + sum_{p=1}^{P} sum_{q=1}^{Q} [
              d_{p,q} * cos(p*theta) * cos(q*alpha)
            + e_{p,q} * cos(p*theta) * sin(q*alpha)
          ]

Basis choices:
- theta in [0, pi]: cosine series (Fourier-cosine, no boundary artifacts)
- alpha in [0, 2*pi): sin+cos trigonometric series (circular, natural periodicity)

Total features: 1 + P + 2Q + 2PQ
For P=5, Q=6: 1 + 5 + 12 + 60 = 78 features per component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------

def _build_design_matrix(
    theta_vals: np.ndarray,
    alpha_vals: np.ndarray,
    n_theta: int,
    n_alpha: int,
) -> np.ndarray:
    """Build (N, n_features) design matrix for 2D Fourier fit.

    Features are ordered as:
        1, cos(theta), ..., cos(P*theta),
        cos(alpha), sin(alpha), ..., cos(Q*alpha), sin(Q*alpha),
        cos(theta)*cos(alpha), cos(theta)*sin(alpha), ...,
        cos(P*theta)*cos(Q*alpha), cos(P*theta)*sin(Q*alpha)
    """
    features = [np.ones(len(theta_vals))]

    # theta-only terms
    for p in range(1, n_theta + 1):
        features.append(np.cos(p * theta_vals))

    # alpha-only terms
    for q in range(1, n_alpha + 1):
        features.append(np.cos(q * alpha_vals))
        features.append(np.sin(q * alpha_vals))

    # cross terms
    for p in range(1, n_theta + 1):
        cp = np.cos(p * theta_vals)
        for q in range(1, n_alpha + 1):
            features.append(cp * np.cos(q * alpha_vals))
            features.append(cp * np.sin(q * alpha_vals))

    return np.column_stack(features)  # (N, n_features)


def _eval_design_row(theta: float, alpha: float, n_theta: int, n_alpha: int) -> np.ndarray:
    """Build feature vector for a single (theta, alpha) point."""
    feats = [1.0]

    for p in range(1, n_theta + 1):
        feats.append(float(np.cos(p * theta)))

    for q in range(1, n_alpha + 1):
        feats.append(float(np.cos(q * alpha)))
        feats.append(float(np.sin(q * alpha)))

    for p in range(1, n_theta + 1):
        cp = float(np.cos(p * theta))
        for q in range(1, n_alpha + 1):
            feats.append(cp * float(np.cos(q * alpha)))
            feats.append(cp * float(np.sin(q * alpha)))

    return np.array(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# FitResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """2D Fourier fit result for PCA amplitude functions A_j(theta, alpha)."""

    m: int                    # number of components fitted
    n_theta: int              # P: theta Fourier harmonics
    n_alpha: int              # Q: alpha Fourier harmonics
    coefficients: np.ndarray  # (m, n_features) fit coefficients
    rmse: np.ndarray          # (m,) per-component RMSE
    r2: np.ndarray            # (m,) per-component R²

    def predict(self, theta: float, alpha: float) -> np.ndarray:
        """Evaluate all m amplitudes at a single (theta, alpha) point.

        Returns:
            (m,) array of predicted amplitudes
        """
        x = _eval_design_row(theta, alpha, self.n_theta, self.n_alpha)
        return self.coefficients @ x  # (m,)

    def predict_batch(
        self, theta_vals: np.ndarray, alpha_vals: np.ndarray
    ) -> np.ndarray:
        """Evaluate all m amplitudes at N (theta, alpha) points.

        Args:
            theta_vals: (N,) array of polar angles
            alpha_vals: (N,) array of rotation angles

        Returns:
            (N, m) array of predicted amplitudes
        """
        D = _build_design_matrix(theta_vals, alpha_vals, self.n_theta, self.n_alpha)
        return D @ self.coefficients.T  # (N, m)


# ---------------------------------------------------------------------------
# Fitting function
# ---------------------------------------------------------------------------

def fit_2d_amplitudes(
    pca_result,
    n_theta: int = 5,
    n_alpha: int = 6,
) -> FitResult:
    """Fit 2D Fourier series to each amplitude function A_j(theta, alpha).

    Only fits the first m = pca_result.m effective components.

    Args:
        pca_result: PCAResult with amplitudes of shape (N_theta, N_alpha, K)
        n_theta:    number of Fourier harmonics for theta (cosine series)
        n_alpha:    number of Fourier harmonics for alpha (sin+cos series)

    Returns:
        FitResult with coefficients, RMSE, and R² per component
    """
    N_theta, N_alpha, K = pca_result.amplitudes.shape
    m = pca_result.m  # fit only the effective components

    # Flatten grid to 1D arrays of (theta, alpha) pairs
    TH, AL = np.meshgrid(pca_result.theta_grid, pca_result.alpha_grid, indexing="ij")
    theta_flat = TH.ravel()  # (N,)
    alpha_flat = AL.ravel()  # (N,)
    N = len(theta_flat)

    # Build design matrix
    D = _build_design_matrix(theta_flat, alpha_flat, n_theta, n_alpha)  # (N, n_features)
    n_features = D.shape[1]

    # Flatten amplitudes to (N, m)
    A_flat = pca_result.amplitudes[:, :, :m].reshape(N, m)

    # Solve m independent least-squares problems
    coefficients = np.zeros((m, n_features))
    rmse = np.zeros(m)
    r2 = np.zeros(m)

    for j in range(m):
        y = A_flat[:, j]
        coeffs, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
        coefficients[j] = coeffs

        y_pred = D @ coeffs
        residuals = y - y_pred
        rmse[j] = float(np.sqrt(np.mean(residuals ** 2)))

        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum(residuals ** 2))
        r2[j] = 1.0 - ss_res / (ss_tot + 1e-30)

    return FitResult(
        m=m,
        n_theta=n_theta,
        n_alpha=n_alpha,
        coefficients=coefficients,
        rmse=rmse,
        r2=r2,
    )
