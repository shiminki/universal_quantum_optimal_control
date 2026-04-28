"""Fidelity contour, pulse-parameter, and fidelity-vs-disorder plots.

All plotting helpers depend only on the public `uqoc` API (propagator, fidelity,
error samplers) — no cross-imports from training scripts.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from uqoc.errors import ERROR_SAMPLERS
from uqoc.fidelity import fidelity
from uqoc.propagator import batched_unitary_generator


def _mc_fidelity(U_target: torch.Tensor, pulse: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
    """Fidelity of a single (U_target, pulse) replicated across error samples."""
    M = errors.shape[1]
    U_rep = U_target.expand(M, -1, -1)
    pulse_rep = pulse.expand(M, -1, -1)
    U_out = batched_unitary_generator(pulse_rep, errors)
    return fidelity(U_out, U_rep, 1)


def fidelity_contour_plot(target_name: str, U_target: torch.Tensor, pulse: torch.Tensor,
                          name: str, save_dir: str, M: int = 10000, phase_only: bool = True,
                          Omega: Optional[float] = None, target_title: Optional[str] = None) -> None:
    """Contour of F over (δ, ε) with a title reporting MC-sampled E[F]."""
    display_name = target_title or target_name
    total_time = float(pulse[:, -1].to(torch.float64).sum().item()) / math.pi

    err_mc = ERROR_SAMPLERS["ore_ple"](M, delta_std=1.0, epsilon_std=0.05)
    F = _mc_fidelity(U_target, pulse, err_mc)
    F_mean = F.mean().item()
    F_err = F.std().item() / math.sqrt(M)

    ORE = torch.linspace(-3, 3, 1000)
    PLE = torch.linspace(-0.15, 0.15, 50)
    ORE_grid, PLE_grid = torch.meshgrid(ORE, PLE, indexing="ij")
    errors_grid = torch.stack([ORE_grid.flatten(), PLE_grid.flatten()], dim=0)
    F_grid = _mc_fidelity(U_target, pulse, errors_grid).reshape(1000, 50)

    ORE_np = ORE_grid.detach().cpu().numpy()
    PLE_np = PLE_grid.detach().cpu().numpy()
    F_np = F_grid.numpy()

    title = (
        rf"{display_name} of {name}" + "\n"
        rf"$\mathbb{{E}}[F] = {F_mean:.4f} \pm {F_err:.4f}$" + "\n"
        rf"Total Evolution Time: ${total_time:.2f}\pi$"
    )
    x_label = r"$\delta / \Omega_{\max} \sim \mathcal{N}(0, 1)$"
    y_label = r"$\epsilon / \Omega_{\max} \sim \mathcal{N}(0, 0.05^2)$"

    if Omega is not None:
        omega_ang = 2 * math.pi * Omega
        total_time = float(pulse[:, -1].to(torch.float64).sum().item()) / omega_ang
        title = (
            rf"{display_name}" + "\n"
            rf"$\mathbb{{E}}[F] = {F_mean:.4f} \pm {F_err:.4f}$" + "\n"
            rf"Total Evolution Time: ${total_time*1000:.2f}$ ns (${total_time*Omega*2:.2f}\pi$)"
            rf" with $\Omega$ = ${Omega}$ (2$\pi)$ MHz"
        )
        ORE_np *= Omega
        PLE_np *= Omega
        x_label = r"$\delta$ (2$\pi$ MHz)"
        y_label = r"$\epsilon$"

    plt.figure(figsize=(18, 14))
    contour = plt.contourf(ORE_np, PLE_np, F_np,
                           levels=[0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0], cmap='viridis')
    plt.contour(ORE_np, PLE_np, F_np, levels=[0.95, 0.99, 0.999], colors='white', linewidths=1.5)
    cbar = plt.colorbar(contour)
    cbar.set_label('Fidelity', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    plt.title(title, fontsize=32)
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{target_name}.png"))
    plt.close()


def get_avg_fidelity(U_target: torch.Tensor, pulse: torch.Tensor, M: int = 10000,
                     delta_list: Optional[List[float]] = None) -> Dict[float, Tuple[float, float]]:
    """Map δ_std → (mean fidelity, stderr) over MC samples."""
    if delta_list is None:
        delta_list = [0.1 * (i + 1) for i in range(10)]
    out: Dict[float, Tuple[float, float]] = {}
    for delta_std in tqdm(delta_list, desc="fidelity-by-std"):
        err = ERROR_SAMPLERS["ore_ple"](M, delta_std=float(delta_std), epsilon_std=0.05)
        F = _mc_fidelity(U_target, pulse, err)
        out[float(delta_std)] = (F.mean().item(), F.std().item() / math.sqrt(M))
    return out


def plot_fidelity_by_std(target_name: str, U_target: torch.Tensor, pulse: torch.Tensor,
                        name: str, save_dir: str, M: int = 10000,
                        target_title: Optional[str] = None) -> None:
    try:
        import pwlf
    except ImportError as e:
        raise ImportError("plot_fidelity_by_std needs `pip install uqoc[viz]` (pwlf)") from e

    display_name = target_title or target_name
    total_time = float(pulse[:, -1].to(torch.float64).sum().item()) / math.pi
    fids = get_avg_fidelity(U_target, pulse, M=M, delta_list=list(torch.arange(0.01, 2.0, 0.01)))

    delta_vals = list(fids.keys())
    F_means = [fids[d][0] for d in delta_vals]
    F_errs = [fids[d][1] for d in delta_vals]
    inF_means = [1 - v for v in F_means]

    os.makedirs(save_dir, exist_ok=True)

    # fidelity vs std(delta) with 2-segment PWLF
    m = pwlf.PiecewiseLinFit(delta_vals, F_means)
    breaks = m.fit(2)
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, F_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, m.predict(delta_vals), 'r--', label='PWLF (2 segments)')
    for i in range(2):
        mid_x = (breaks[i] + breaks[i + 1]) / 2
        mid_y = m.predict([mid_x])[0]
        plt.text(mid_x, mid_y - 0.03, f"y = {m.slopes[i]:.3f}x + {m.intercepts[i]:.3f}",
                 ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.xlabel(r"Std$(\delta / \Omega_{\max})$", fontsize=18)
    plt.ylabel("Expected Fidelity", fontsize=18)
    plt.title(rf"Fidelity curve for {display_name}" + "\n" + rf"Total Evolution Time: ${total_time:.2f}\pi$", fontsize=20)
    plt.ylim(0.6, 1); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{target_name}_{name}_fidelity.png"))
    plt.close()

    # log-log infidelity vs std(delta) with 3-segment PWLF
    log_d = np.log(delta_vals)
    log_iF = np.log(inF_means)
    m2 = pwlf.PiecewiseLinFit(log_d, log_iF)
    breaks2 = m2.fit(3)
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, inF_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, np.exp(m2.predict(log_d)), 'r--', label='PWLF (3 segments)')
    plt.semilogy(); plt.semilogx()
    for i in range(3):
        mid_x_log = (breaks2[i] + breaks2[i + 1]) / 2
        mid_x = np.exp(mid_x_log)
        mid_y = np.exp(m2.predict([mid_x_log])[0])
        plt.text(mid_x, mid_y * 1.2, f"log(y) = {m2.slopes[i]:.3f} log(x) + {m2.intercepts[i]:.3f}",
                 ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.xlabel(r"Std$(\delta / \Omega_{\max})$"); plt.ylabel("Expected Infidelity")
    plt.title(rf"Infidelity curve for {display_name} of {name}" + "\n" + rf"Total Evolution Time: ${total_time:.2f}\pi$")
    plt.ylim(1e-3, 1); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{target_name}_infidelity_with_fit.png"))
    plt.close()


def plot_pulse_param(file_path: str, title: str, df, Omega: float, df_smooth=None,
                     phi: float = 0.0, theta: float = 0.0, lam: float = 0.0,
                     model: str = "", cutoff: float = 0.0) -> None:
    """Step-plot of (Ω_x, Ω_y, Ω_z) over time; optionally overlay a smoothed version.

    The DataFrame is expected to carry columns:
        t (us), Omega_x (2pi MHz), Omega_y (2pi MHz), Omega_z (2pi MHz)
    (as produced by app.py's `csv_to_internal_pulse` step).
    """
    t = df["t (us)"].to_numpy(dtype=float) * 1000
    H_x = df["Omega_x (2pi MHz)"].to_numpy(dtype=float)
    H_y = df["Omega_y (2pi MHz)"].to_numpy(dtype=float)
    H_z = df["Omega_z (2pi MHz)"].to_numpy(dtype=float)
    t_edges = np.concatenate(([0.0], np.cumsum(t)))
    total_time = t.sum()
    labels = [r"$\Omega_x(t)$ $(2\pi$ MHz)", r"$\Omega_y(t)$ $(2\pi$ MHz)", r"$\Omega_z(t)$ $(2\pi$ MHz)"]
    base_title = (
        rf"Target = $U[\vec{{n}}(\phi={phi:.3f}, \theta={theta:.3f}), \alpha={lam:.3f}\pi]$" + "\n"
        rf"Rabi = $(2\pi)\ {Omega}$ MHz, runtime = {total_time:.2f} ns"
    )

    if df_smooth is not None:
        Hx_s = df_smooth["Omega_x (2pi MHz)"].to_numpy(dtype=float)
        Hy_s = df_smooth["Omega_y (2pi MHz)"].to_numpy(dtype=float)
        Hz_s = df_smooth["Omega_z (2pi MHz)"].to_numpy(dtype=float)
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)
        for i, arr in enumerate([H_x, H_y, H_z]):
            axes[i].step(t_edges, np.r_[arr, arr[-1]], where='post', color='C0')
            axes[i].set_ylabel(labels[i], fontsize=16); axes[i].grid(True)
            axes[i].set_ylim(-1.1 * Omega, 1.1 * Omega)
        axes[0].set_title(base_title, fontsize=24)
        for i, arr in enumerate([Hx_s, Hy_s, Hz_s]):
            axes[3 + i].step(t_edges, np.r_[arr, arr[-1]], where='post', color='C1')
            axes[3 + i].set_ylabel(labels[i], fontsize=16); axes[3 + i].grid(True)
            axes[3 + i].set_ylim(-1.1 * Omega, 1.1 * Omega)
        axes[3].set_title(rf"Low pass filter with cutoff $(2\pi)\ {cutoff:.0f}$ MHz", fontsize=24)
        axes[5].set_xlabel("Rotation time (ns)", fontsize=18)
    else:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 6), sharex=True)
        for i, arr in enumerate([H_x, H_y, H_z]):
            axes[i].step(t_edges, np.r_[arr, arr[-1]], where='post', color='C0')
            axes[i].set_ylabel(labels[i], fontsize=18); axes[i].grid(True)
            axes[i].set_ylim(-1.1 * Omega, 1.1 * Omega)
        axes[0].set_title(base_title, fontsize=24)
        axes[2].set_xlabel("Rotation time (ns)", fontsize=18)

    os.makedirs(file_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f"{title}.png"))
    plt.close(fig)
