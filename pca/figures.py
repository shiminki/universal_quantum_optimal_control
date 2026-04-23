"""Publication-quality figures for PCA analysis of transformer control pulses.

Six figures, each making a specific scientific argument about the low-dimensional
structure of the transformer output manifold.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
})

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ---------------------------------------------------------------------------
# Figure 1: Singular Value Spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    pca_phi,
    pca_tau,
    pca_full,
    out_path: str,
    n_show: int = 30,
) -> None:
    """Figure 1: Three-panel singular value analysis.

    (a) Normalized singular values on log scale — reveals spectral cliff.
    (b) Cumulative explained variance — shows where 99.9% threshold is crossed.
    (c) Singular value ratios sigma_k / sigma_{k+1} — identifies spectral gaps.

    Scientific argument: If sigma collapses by orders of magnitude after m
    components, the manifold is provably low-dimensional.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pca_list = [(pca_phi, "φ (phase)", "tab:blue"),
                (pca_tau, "τ (duration)", "tab:orange"),
                (pca_full, "full [φ, τ]", "tab:green")]

    # --- (a) Normalized singular values ---
    ax = axes[0]
    for pca, label, color in pca_list:
        k = min(n_show, len(pca.S))
        ks = np.arange(1, k + 1)
        ax.semilogy(ks, pca.S[:k] / pca.S[0], "o-", color=color,
                    label=label, linewidth=1.5, markersize=4)
        if pca.m <= n_show:
            ax.axvline(pca.m, color=color, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(pca.m + 0.3, pca.S[pca.m - 1] / pca.S[0] * 1.5,
                    f"m={pca.m}", color=color, fontsize=9)
    ax.set_xlabel("Component index k")
    ax.set_ylabel(r"Normalized singular value  $\sigma_k / \sigma_1$")
    ax.set_title("(a) Singular value spectrum")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, n_show + 0.5])

    # --- (b) Cumulative explained variance ---
    ax = axes[1]
    for pca, label, color in pca_list:
        k = min(n_show, len(pca.evr))
        ks = np.arange(1, k + 1)
        cumvar = np.cumsum(pca.evr[:k]) * 100
        ax.plot(ks, cumvar, "o-", color=color,
                label=f"{label}  (m={pca.m})", linewidth=1.5, markersize=4)
        if pca.m <= n_show:
            ax.axvline(pca.m, color=color, linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(99.9, color="red", linestyle=":", linewidth=1.5, label="99.9%")
    ax.set_xlabel("Component index k")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("(b) Cumulative variance")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, n_show + 0.5])
    ax.set_ylim([max(0, np.cumsum(pca_full.evr[:1])[0] * 100 - 5), 101])

    # --- (c) Spectral gaps ---
    ax = axes[2]
    for pca, label, color in pca_list:
        k = min(n_show, len(pca.S) - 1)
        ks = np.arange(1, k + 1)
        gaps = pca.S[:k] / (pca.S[1 : k + 1] + 1e-30)
        ax.semilogy(ks, gaps, "o-", color=color, label=label,
                    linewidth=1.5, markersize=4)
    ax.set_xlabel("Component index k")
    ax.set_ylabel(r"Singular value ratio  $\sigma_k / \sigma_{k+1}$")
    ax.set_title("(c) Spectral gaps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, n_show + 0.5])

    fig.suptitle(
        "PCA spectrum of transformer control pulses  "
        r"$\varphi(t;\,\theta,\alpha)$",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: PCA Basis Functions (mean pulse + top PC vectors)
# ---------------------------------------------------------------------------

def plot_basis(
    pca_full,
    out_path: str,
    L: int = 400,
    n_show: Optional[int] = None,
) -> None:
    """Figure 2: Mean pulse and top principal component vectors.

    Row 0: mean control sequence (phi and tau separately).
    Rows 1..m: j-th PC vector — cos(phi), sin(phi), and tau components.

    Scientific argument: Structured, spatially coherent PC modes indicate
    the transformer found a smooth, organized parameterization of SU(2).
    """
    if n_show is None:
        n_show = pca_full.m
    n_show = min(n_show, 8, pca_full.Vt.shape[0])

    t_idx = np.arange(L)
    n_rows = n_show + 1

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis]

    # --- Row 0: mean pulse ---
    mean_cos_phi = pca_full.X_mean[:L]
    mean_sin_phi = pca_full.X_mean[L : 2 * L]
    mean_tau_norm = pca_full.X_mean[2 * L : 3 * L]
    mean_phi = np.arctan2(mean_sin_phi, mean_cos_phi)

    axes[0, 0].plot(t_idx, mean_phi, linewidth=0.8, color="steelblue")
    axes[0, 0].set_ylabel("φ̄ (rad)")
    axes[0, 0].set_title("Mean phase  φ̄(t)", fontsize=11)
    axes[0, 0].set_ylim([-math.pi - 0.3, math.pi + 0.3])

    axes[0, 1].plot(t_idx, mean_tau_norm, linewidth=0.8, color="darkorange")
    axes[0, 1].set_ylabel("τ̄ (normalized)")
    axes[0, 1].set_title("Mean duration  τ̄(t)", fontsize=11)

    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5, f"Mean pulse\n(averaged over all gates)\nL = {L}",
                    ha="center", va="center", fontsize=11,
                    transform=axes[0, 2].transAxes)

    # --- Rows 1..n_show: PC vectors ---
    for row_idx in range(1, n_show + 1):
        j = row_idx - 1
        V = pca_full.Vt[j]  # (3L,)
        V_cos = V[:L]
        V_sin = V[L : 2 * L]
        V_tau = V[2 * L : 3 * L]
        evr_pct = pca_full.evr[j] * 100

        axes[row_idx, 0].plot(t_idx, V_cos, linewidth=0.8, color="steelblue")
        axes[row_idx, 0].set_ylabel("cos-φ component")
        axes[row_idx, 0].set_title(
            f"PC {j + 1}  cos-φ  (EVR = {evr_pct:.2f}%)", fontsize=11
        )

        axes[row_idx, 1].plot(t_idx, V_sin, linewidth=0.8, color="seagreen")
        axes[row_idx, 1].set_ylabel("sin-φ component")
        axes[row_idx, 1].set_title(f"PC {j + 1}  sin-φ", fontsize=11)

        axes[row_idx, 2].plot(t_idx, V_tau, linewidth=0.8, color="darkorange")
        axes[row_idx, 2].set_ylabel("τ component")
        axes[row_idx, 2].set_title(f"PC {j + 1}  τ", fontsize=11)

    for row in axes:
        for ax in row:
            ax.set_xlabel("Pulse index i")
            ax.grid(True, alpha=0.2)

    fig.suptitle("PCA basis functions of transformer control pulses", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: 2D Amplitude Heatmaps
# ---------------------------------------------------------------------------

def plot_amplitudes(
    pca_result,
    fit_result,
    out_path: str,
    n_fine: int = 80,
) -> None:
    """Figure 3: Amplitude maps A_j(theta, alpha) — data vs. Fourier fit.

    Left column: raw PCA amplitudes on the training grid.
    Right column: Fourier fit evaluated on a fine grid.

    Scientific argument: If A_j is smooth and the Fourier fit achieves
    low RMSE, the entire 800-parameter output is determined by
    m × (P+1)(2Q+1) Fourier coefficients — a dramatic compression.
    """
    m = fit_result.m
    theta_g = pca_result.theta_grid
    alpha_g = pca_result.alpha_grid

    # Fine grid for fit visualization
    theta_fine = np.linspace(theta_g[0], theta_g[-1], n_fine)
    alpha_fine = np.linspace(alpha_g[0], alpha_g[-1], n_fine)
    TH_fine, AL_fine = np.meshgrid(theta_fine, alpha_fine, indexing="ij")
    A_fine_flat = fit_result.predict_batch(
        TH_fine.ravel(), AL_fine.ravel()
    )  # (n_fine^2, m)
    A_fine = A_fine_flat.reshape(n_fine, n_fine, m)

    fig, axes = plt.subplots(m, 2, figsize=(12, 4 * m))
    if m == 1:
        axes = axes[np.newaxis]

    extent = [
        alpha_g[0] / (2 * math.pi), alpha_g[-1] / (2 * math.pi),
        theta_g[0] / math.pi,       theta_g[-1] / math.pi,
    ]
    extent_fine = [
        alpha_fine[0] / (2 * math.pi), alpha_fine[-1] / (2 * math.pi),
        theta_fine[0] / math.pi,       theta_fine[-1] / math.pi,
    ]

    for j in range(m):
        A_raw = pca_result.amplitudes[:, :, j]  # (N_theta, N_alpha)
        vmax = max(float(np.abs(A_raw).max()), float(np.abs(A_fine[:, :, j]).max()))
        vmin = -vmax

        # Left: raw data
        im0 = axes[j, 0].imshow(
            A_raw,
            origin="lower", aspect="auto", extent=extent,
            vmin=vmin, vmax=vmax, cmap="RdBu_r",
        )
        axes[j, 0].set_xlabel("α / 2π")
        axes[j, 0].set_ylabel("θ / π")
        axes[j, 0].set_title(
            f"PC {j + 1}: A_{j + 1}(θ, α)  — training data  "
            f"(EVR = {pca_result.evr[j] * 100:.2f}%)",
            fontsize=11,
        )
        plt.colorbar(im0, ax=axes[j, 0], fraction=0.046, pad=0.04)

        # Right: Fourier fit
        im1 = axes[j, 1].imshow(
            A_fine[:, :, j],
            origin="lower", aspect="auto", extent=extent_fine,
            vmin=vmin, vmax=vmax, cmap="RdBu_r",
        )
        axes[j, 1].set_xlabel("α / 2π")
        axes[j, 1].set_ylabel("θ / π")
        axes[j, 1].set_title(
            f"PC {j + 1}: Fourier fit  "
            f"(RMSE = {fit_result.rmse[j]:.3f},  R² = {fit_result.r2[j]:.4f})",
            fontsize=11,
        )
        plt.colorbar(im1, ax=axes[j, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        r"Amplitude functions  $A_j(\theta, \alpha)$  of PCA modes",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Reconstruction Fidelity Map
# ---------------------------------------------------------------------------

def plot_fidelity_map(
    theta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    model_fid_map: np.ndarray,
    recon_fid_map: np.ndarray,
    out_path: str,
) -> None:
    """Figure 4: Model vs. reconstruction fidelity over the (theta, alpha) plane.

    (a) Model fidelity F(theta, alpha) — establishes transformer baseline.
    (b) Reconstruction fidelity F_recon(theta, alpha) — PCA formula quality.
    (c) Fidelity degradation F_model - F_recon — where compression loses info.

    Scientific argument: If reconstruction fidelity stays > 0.999 while
    the model achieves > 0.9999, the PCA + Fourier formula is nearly lossless.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    extent = [
        alpha_grid[0] / (2 * math.pi), alpha_grid[-1] / (2 * math.pi),
        theta_grid[0] / math.pi,       theta_grid[-1] / math.pi,
    ]
    imshow_kw = dict(origin="lower", aspect="auto", extent=extent)

    # Shared fidelity color range
    fid_min = max(0.9, min(float(model_fid_map.min()), float(recon_fid_map.min())) - 0.01)
    fid_max = 1.0

    im0 = axes[0].imshow(model_fid_map, vmin=fid_min, vmax=fid_max,
                         cmap="viridis", **imshow_kw)
    axes[0].set_title(
        f"(a) Model fidelity\n(mean = {model_fid_map.mean():.5f})", fontsize=11
    )
    axes[0].set_xlabel("α / 2π"); axes[0].set_ylabel("θ / π")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon_fid_map, vmin=fid_min, vmax=fid_max,
                         cmap="viridis", **imshow_kw)
    axes[1].set_title(
        f"(b) PCA reconstruction fidelity\n(mean = {recon_fid_map.mean():.5f})", fontsize=11
    )
    axes[1].set_xlabel("α / 2π"); axes[1].set_ylabel("θ / π")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    degradation = model_fid_map - recon_fid_map
    dmax = float(max(abs(degradation).max(), 1e-6))
    im2 = axes[2].imshow(degradation, vmin=-dmax, vmax=dmax,
                         cmap="RdBu_r", **imshow_kw)
    axes[2].set_title(
        f"(c) Fidelity degradation  (model − recon)\n"
        f"max = {float(degradation.max()):.5f},  "
        f"mean = {float(degradation.mean()):.5f}",
        fontsize=11,
    )
    axes[2].set_xlabel("α / 2π"); axes[2].set_ylabel("θ / π")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Reconstruction fidelity across the gate space  "
        r"$(\theta, \alpha)$",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Ablation — Fidelity vs. Number of PCA Components
# ---------------------------------------------------------------------------

def plot_ablation(
    n_components_list: List[int],
    mean_fid_list: List[float],
    l2_errors_list: List[float],
    out_path: str,
) -> None:
    """Figure 5: How reconstruction quality depends on number of PCA components.

    Left y-axis: mean zero-error fidelity over test grid.
    Right y-axis: mean L2 error in phi space.

    Scientific argument: The "information knee" — fidelity plateau at k = m
    directly quantifies the intrinsic dimension of the control manifold.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Number of PCA components", fontsize=13)
    ax1.set_ylabel("Mean zero-error fidelity", color=color1, fontsize=13)
    line1, = ax1.plot(n_components_list, mean_fid_list, "o-",
                      color=color1, linewidth=2.0, markersize=7, label="Fidelity")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0.999, color="red", linestyle=":", linewidth=1.5, label="F = 0.999")
    ax1.set_ylim([
        max(0.0, min(mean_fid_list) - 0.05),
        min(1.005, max(mean_fid_list) + 0.005),
    ])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Mean L2 error in φ (rad)", color=color2, fontsize=13)
    line2, = ax2.plot(n_components_list, l2_errors_list, "s--",
                      color=color2, linewidth=2.0, markersize=7, label="L2 error")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11)

    plt.title(
        "Reconstruction quality vs. number of PCA components\n"
        r"(test grid, zero-error fidelity)",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: 1D Slices of Amplitude Functions
# ---------------------------------------------------------------------------

def plot_slices(
    pca_result,
    fit_result,
    out_path: str,
    n_alpha_slices: int = 3,
    n_theta_slices: int = 3,
) -> None:
    """Figure 6: 1D slices of A_j(theta, alpha) for fixed alpha or theta.

    Left column: A_j vs theta at 3 fixed alpha values.
    Right column: A_j vs alpha at 3 fixed theta values.
    Dots = data, lines = Fourier fit.

    Scientific argument: Validates the 2D Fourier fit and reveals
    whether the theta/alpha dependence is smooth and nearly separable.
    """
    m = fit_result.m
    theta_g = pca_result.theta_grid
    alpha_g = pca_result.alpha_grid

    alpha_idxs = np.round(np.linspace(0, len(alpha_g) - 1, n_alpha_slices)).astype(int)
    theta_idxs = np.round(np.linspace(0, len(theta_g) - 1, n_theta_slices)).astype(int)

    fig, axes = plt.subplots(m, 2, figsize=(14, 3.8 * m))
    if m == 1:
        axes = axes[np.newaxis]

    for j in range(m):
        A_data = pca_result.amplitudes[:, :, j]  # (N_theta, N_alpha)

        # --- Left: A_j vs theta for fixed alpha values ---
        ax = axes[j, 0]
        for ci, ai in enumerate(alpha_idxs):
            alpha_fixed = float(alpha_g[ai])
            A_slice = A_data[:, ai]

            theta_dense = np.linspace(theta_g[0], theta_g[-1], 200)
            A_fit = fit_result.predict_batch(
                theta_dense, np.full_like(theta_dense, alpha_fixed)
            )[:, j]

            c = _COLORS[ci % len(_COLORS)]
            ax.plot(theta_g / math.pi, A_slice, "o", color=c,
                    markersize=5, alpha=0.7)
            ax.plot(theta_dense / math.pi, A_fit, "-", color=c, linewidth=1.5,
                    label=f"α = {alpha_fixed / (2 * math.pi):.2f}·2π")

        ax.set_xlabel("θ / π")
        ax.set_ylabel(f"$A_{{{j + 1}}}(\\theta, \\alpha)$")
        ax.set_title(f"PC {j + 1}: fixed α slices", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Right: A_j vs alpha for fixed theta values ---
        ax = axes[j, 1]
        for ci, ti in enumerate(theta_idxs):
            theta_fixed = float(theta_g[ti])
            A_slice = A_data[ti, :]

            alpha_dense = np.linspace(alpha_g[0], alpha_g[-1], 200)
            A_fit = fit_result.predict_batch(
                np.full_like(alpha_dense, theta_fixed), alpha_dense
            )[:, j]

            c = _COLORS[ci % len(_COLORS)]
            ax.plot(alpha_g / (2 * math.pi), A_slice, "o", color=c,
                    markersize=5, alpha=0.7)
            ax.plot(alpha_dense / (2 * math.pi), A_fit, "-", color=c, linewidth=1.5,
                    label=f"θ = {theta_fixed / math.pi:.2f}·π")

        ax.set_xlabel("α / 2π")
        ax.set_ylabel(f"$A_{{{j + 1}}}(\\theta, \\alpha)$")
        ax.set_title(f"PC {j + 1}: fixed θ slices", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"1D slices of amplitude functions  $A_j(\theta, \alpha)$  "
        "— data (dots) vs. Fourier fit (lines)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
