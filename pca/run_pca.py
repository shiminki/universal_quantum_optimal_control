"""CLI entry point for PCA analysis of transformer control pulses.

Usage (from project root):
    python -m pca.run_pca
    python -m pca.run_pca --model_size 400 --N_theta 40 --N_alpha 50
    python -m pca.run_pca --out_dir pca/results --n_theta_fourier 5 --n_alpha_fourier 6

Output structure:
    pca/results/
        data.npz           — raw inference output + encoded matrices
        pca_results.npz    — PCA decompositions for phi, tau, full
        figures/
            fig1_spectrum.png
            fig2_basis.png
            fig3_amplitudes.png
            fig4_fidelity_map.png
            fig5_ablation.png
            fig6_slices.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

# Add project root to path so local imports work when run as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pca.sample import load_pipeline, build_sampling_grid, build_data_matrix
from pca.decompose import run_pca
from pca.fit import fit_2d_amplitudes
from pca.reconstruct import (
    PulseReconstructor,
    eval_zero_error_fidelity,
    validate_reconstruction,
)
from pca.figures import (
    plot_spectrum,
    plot_basis,
    plot_amplitudes,
    plot_fidelity_map,
    plot_ablation,
    plot_slices,
)


# ---------------------------------------------------------------------------
# Helper: batch fidelity evaluation
# ---------------------------------------------------------------------------

def _batch_fidelity(
    pulses: np.ndarray,
    theta_vals: np.ndarray,
    alpha_vals: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute zero-error fidelity for many pulses in batches."""
    return eval_zero_error_fidelity(pulses, theta_vals, alpha_vals, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA analysis of transformer control pulses phi(t; theta, alpha)"
    )
    parser.add_argument("--model_size", type=int, default=400,
                        help="Transformer model size: 400 (default) or 100")
    parser.add_argument("--N_theta", type=int, default=40,
                        help="Number of polar angle grid points (default 40)")
    parser.add_argument("--N_alpha", type=int, default=50,
                        help="Number of rotation angle grid points (default 50)")
    parser.add_argument("--out_dir", type=str, default="pca/results",
                        help="Output directory (default pca/results)")
    parser.add_argument("--n_theta_fourier", type=int, default=5,
                        help="Fourier harmonics for theta fit (default 5)")
    parser.add_argument("--n_alpha_fourier", type=int, default=6,
                        help="Fourier harmonics for alpha fit (default 6)")
    parser.add_argument("--energy_threshold", type=float, default=0.999,
                        help="Cumulative variance threshold for effective dimension (default 0.999)")
    parser.add_argument("--max_components", type=int, default=50,
                        help="Maximum PCA components to store (default 50)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Inference batch size (default 256)")
    parser.add_argument("--ablation_max", type=int, default=15,
                        help="Maximum components for ablation study (default 15)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    N = args.N_theta * args.N_alpha
    print(f"\n{'='*60}")
    print(f"PCA Analysis: {args.N_theta} × {args.N_alpha} = {N} grid points")
    print(f"Model: length_{args.model_size}.pt")
    print(f"Output: {args.out_dir}/")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/7] Loading model...")
    pipeline, tau_min, tau_max = load_pipeline(args.model_size)
    L = args.model_size  # number of pulses

    # ------------------------------------------------------------------
    # 2. Build grid and infer pulses
    # ------------------------------------------------------------------
    print(f"\n[2/7] Sampling {args.N_theta}×{args.N_alpha} grid and running inference...")
    theta_grid, alpha_grid, rotation_vecs = build_sampling_grid(
        N_theta=args.N_theta, N_alpha=args.N_alpha
    )
    raw_pulses, X_phi, X_tau, X_full = build_data_matrix(
        pipeline, rotation_vecs, batch_size=args.batch_size,
        tau_min=tau_min, tau_max=tau_max,
    )
    print(f"  raw_pulses: {raw_pulses.shape}   X_full: {X_full.shape}")

    # Flatten theta/alpha grids matching raw_pulses order
    TH, AL = np.meshgrid(theta_grid, alpha_grid, indexing="ij")
    theta_flat = TH.ravel()
    alpha_flat = AL.ravel()

    # Save raw data
    np.savez(
        os.path.join(args.out_dir, "data.npz"),
        raw_pulses=raw_pulses,
        X_phi=X_phi, X_tau=X_tau, X_full=X_full,
        theta_grid=theta_grid, alpha_grid=alpha_grid,
        rotation_vecs=rotation_vecs,
    )
    print(f"  Saved data.npz")

    # ------------------------------------------------------------------
    # 3. PCA on all three encodings
    # ------------------------------------------------------------------
    print("\n[3/7] Running PCA...")
    pca_phi  = run_pca(X_phi,  theta_grid, alpha_grid, label="phi",
                       energy_threshold=args.energy_threshold,
                       max_components=args.max_components)
    pca_tau  = run_pca(X_tau,  theta_grid, alpha_grid, label="tau",
                       energy_threshold=args.energy_threshold,
                       max_components=args.max_components)
    pca_full = run_pca(X_full, theta_grid, alpha_grid, label="full",
                       energy_threshold=args.energy_threshold,
                       max_components=args.max_components)

    for pca in [pca_phi, pca_tau, pca_full]:
        cum = float(np.cumsum(pca.evr)[pca.m - 1])
        print(f"  [{pca.label:4s}] m = {pca.m:2d}  "
              f"({cum * 100:.3f}% variance explained)")

    np.savez(
        os.path.join(args.out_dir, "pca_results.npz"),
        phi_S=pca_phi.S,   phi_Vt=pca_phi.Vt,   phi_evr=pca_phi.evr,   phi_m=pca_phi.m,
        tau_S=pca_tau.S,   tau_Vt=pca_tau.Vt,   tau_evr=pca_tau.evr,   tau_m=pca_tau.m,
        full_S=pca_full.S, full_Vt=pca_full.Vt, full_evr=pca_full.evr, full_m=pca_full.m,
        full_X_mean=pca_full.X_mean,
        theta_grid=theta_grid, alpha_grid=alpha_grid,
    )
    print("  Saved pca_results.npz")

    # ------------------------------------------------------------------
    # 4. Fit 2D Fourier amplitudes
    # ------------------------------------------------------------------
    print("\n[4/7] Fitting 2D Fourier amplitudes...")
    fit_full = fit_2d_amplitudes(
        pca_full,
        n_theta=args.n_theta_fourier,
        n_alpha=args.n_alpha_fourier,
    )
    for j in range(fit_full.m):
        print(f"  PC {j + 1}: RMSE = {fit_full.rmse[j]:.4f}   R² = {fit_full.r2[j]:.5f}")

    # ------------------------------------------------------------------
    # 5. Fidelity maps
    # ------------------------------------------------------------------
    print("\n[5/7] Computing fidelity maps...")
    reconstructor = PulseReconstructor(pca_full, fit_full, L=L,
                                       tau_min=tau_min, tau_max=tau_max)

    # Model fidelity (all N training points, zero error)
    print("  Model fidelities...")
    model_fids = _batch_fidelity(raw_pulses, theta_flat, alpha_flat,
                                 batch_size=args.batch_size)
    model_fid_map = model_fids.reshape(args.N_theta, args.N_alpha)
    print(f"  Model:   mean={model_fids.mean():.5f}  min={model_fids.min():.5f}")

    # Reconstruction fidelity (all N points, vectorized)
    print("  Reconstruction fidelities...")
    recon_pulses_all = reconstructor.get_pulses_batch(theta_flat, alpha_flat)  # (N, L, 2)
    recon_fids = _batch_fidelity(recon_pulses_all, theta_flat, alpha_flat,
                                 batch_size=args.batch_size)
    recon_fid_map = recon_fids.reshape(args.N_theta, args.N_alpha)
    print(f"  Recon:   mean={recon_fids.mean():.5f}  min={recon_fids.min():.5f}")
    print(f"  Max degradation: {float((model_fid_map - recon_fid_map).max()):.5f}")

    np.savez(
        os.path.join(args.out_dir, "fidelity_maps.npz"),
        model_fid_map=model_fid_map,
        recon_fid_map=recon_fid_map,
        theta_grid=theta_grid,
        alpha_grid=alpha_grid,
    )
    print("  Saved fidelity_maps.npz")

    # ------------------------------------------------------------------
    # 6. Ablation study
    # ------------------------------------------------------------------
    print("\n[6/7] Ablation study (reconstruction vs. n_components)...")

    # Test grid: midpoints between training grid points → never saw by PCA
    theta_step = (theta_grid[-1] - theta_grid[0]) / max(1, args.N_theta - 1)
    alpha_step = (alpha_grid[-1] - alpha_grid[0]) / max(1, args.N_alpha - 1)
    theta_test_1d = theta_grid[:-1] + theta_step / 2  # midpoints
    alpha_test_1d = alpha_grid[:-1] + alpha_step / 2

    # Take at most 12 points per axis to keep ablation fast
    theta_test_1d = theta_test_1d[::max(1, len(theta_test_1d) // 12)][:12]
    alpha_test_1d = alpha_test_1d[::max(1, len(alpha_test_1d) // 12)][:12]

    TH_test, AL_test = np.meshgrid(theta_test_1d, alpha_test_1d, indexing="ij")
    theta_test_flat = TH_test.ravel()
    alpha_test_flat = AL_test.ravel()
    N_test = len(theta_test_flat)

    # Model pulses for test set
    n_x_t = np.sin(theta_test_flat).astype(np.float32)
    n_z_t = np.cos(theta_test_flat).astype(np.float32)
    rot_test = np.stack([n_x_t, np.zeros_like(n_x_t), n_z_t,
                         alpha_test_flat.astype(np.float32)], axis=1)
    test_model_pulses_list = []
    with torch.no_grad():
        for i in range(0, N_test, args.batch_size):
            b = torch.tensor(rot_test[i : i + args.batch_size], dtype=torch.float32)
            test_model_pulses_list.append(pipeline(b).cpu().numpy())
    test_model_pulses = np.concatenate(test_model_pulses_list, axis=0)  # (N_test, L, 2)

    n_max = min(args.ablation_max, pca_full.Vt.shape[0], fit_full.m)
    n_comp_range = list(range(1, n_max + 1))
    mean_fid_ablation = []
    l2_err_ablation = []

    for n_comp in n_comp_range:
        recon_test = reconstructor.get_pulses_batch(
            theta_test_flat, alpha_test_flat, n_components=n_comp
        )  # (N_test, L, 2)
        fids = _batch_fidelity(recon_test, theta_test_flat, alpha_test_flat,
                               batch_size=256)
        dphi = np.angle(np.exp(1j * (test_model_pulses[:, :, 0] - recon_test[:, :, 0])))
        l2 = float(np.sqrt(np.mean(dphi ** 2)))
        mean_fid_ablation.append(float(fids.mean()))
        l2_err_ablation.append(l2)
        print(f"  n={n_comp:2d}: F = {fids.mean():.5f}   L2(φ) = {l2:.4f} rad")

    # ------------------------------------------------------------------
    # 7. Generate figures
    # ------------------------------------------------------------------
    print("\n[7/7] Generating figures...")

    plot_spectrum(
        pca_phi, pca_tau, pca_full,
        out_path=os.path.join(fig_dir, "fig1_spectrum.png"),
    )
    plot_basis(
        pca_full,
        out_path=os.path.join(fig_dir, "fig2_basis.png"),
        L=L,
    )
    plot_amplitudes(
        pca_full, fit_full,
        out_path=os.path.join(fig_dir, "fig3_amplitudes.png"),
    )
    plot_fidelity_map(
        theta_grid, alpha_grid, model_fid_map, recon_fid_map,
        out_path=os.path.join(fig_dir, "fig4_fidelity_map.png"),
    )
    plot_ablation(
        n_comp_range, mean_fid_ablation, l2_err_ablation,
        out_path=os.path.join(fig_dir, "fig5_ablation.png"),
    )
    plot_slices(
        pca_full, fit_full,
        out_path=os.path.join(fig_dir, "fig6_slices.png"),
    )

    print(f"\n{'='*60}")
    print(f"Done!  Results in: {args.out_dir}/")
    print(f"Figures in:        {fig_dir}/")
    print(f"\nKey findings:")
    print(f"  φ effective dim: m = {pca_phi.m}")
    print(f"  τ effective dim: m = {pca_tau.m}")
    print(f"  Joint eff. dim:  m = {pca_full.m}")
    print(f"  Mean model fidelity:  {model_fids.mean():.5f}")
    print(f"  Mean recon fidelity:  {recon_fids.mean():.5f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
