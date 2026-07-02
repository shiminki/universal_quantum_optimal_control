#!/usr/bin/env python3
"""Publication-quality figures for the thesis.

Usage:
    python figures.py [--figure 1|2|3|all]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_root = Path(__file__).resolve().parent
_src = _root / "src"
for _p in (_src, _root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uqoc.config import load_config
from uqoc.errors import ERROR_SAMPLERS
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import rotation_vector_to_unitary
from smoothing.smooth_pulse import lowpass_filter

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.titlesize": 20,
    "lines.linewidth": 1.5,
})

OUTPUT_DIR = "figures"
OMEGA_MAX = 80.0          # 2π MHz, default Rabi rate
MC_N = 5000               # Monte-Carlo samples for fidelity averages

CONTOUR_LEVELS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0]
CONTOUR_WHITE  = [0.95, 0.99, 0.999]

# Shared gate parameters -------------------------------------------------
# H gate: axis n(φ=0, θ=π/4), rotation = π
_H_X, _H_Y, _H_Z, _H_TH = math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4), 1.0
# X(π/2) gate: axis n(φ=0, θ=π/2), rotation = π/2
_XH_X, _XH_Y, _XH_Z, _XH_TH = 1.0, 0.0, 0.0, 0.5


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _load_pipe(config: str, weights: str) -> Pipeline:
    cfg = load_config(config)
    return Pipeline(build_model(cfg.model), weight_path=weights, device="cpu")


def _get_pulse_and_target(
    pipe: Pipeline,
    x: float, y: float, z: float,
    theta_pi: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = np.array([x, y, z], dtype=float)
    axis /= np.linalg.norm(axis)
    # Nudge rotation away from π singularity (mirrors app.py behaviour)
    theta = 0.99 * math.pi if abs(theta_pi - 1.0) < 1e-9 else math.pi * theta_pi
    rot_vec = torch.tensor([*axis, theta], dtype=torch.float32).unsqueeze(0)
    pulse = pipe(rot_vec).squeeze(0)
    rot_vec_t = torch.tensor([*axis, math.pi * theta_pi], dtype=torch.float32)
    U_target = rotation_vector_to_unitary(rot_vec_t)
    return pulse, U_target


def _mc_fidelity(
    U_target: torch.Tensor,
    pulse: torch.Tensor,
    errors: torch.Tensor,
) -> torch.Tensor:
    """Fidelity of (U_target, pulse) replicated over a batch of error samples."""
    M = errors.shape[1]
    U_out = batched_unitary_generator(pulse.expand(M, -1, -1), errors)
    return fidelity(U_out, U_target.expand(M, -1, -1), 1)



def _contour_data(
    U_target: torch.Tensor,
    pulse: torch.Tensor,
    M: int = MC_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    err_mc = ERROR_SAMPLERS["ore_ple"](M, delta_std=1.0, epsilon_std=0.05)
    F      = _mc_fidelity(U_target, pulse, err_mc)
    F_mean = F.mean().item()
    F_err  = F.std().item() / math.sqrt(M)

    ORE = torch.linspace(-3, 3, 1000)
    PLE = torch.linspace(-0.15, 0.15, 50)
    ORE_g, PLE_g = torch.meshgrid(ORE, PLE, indexing="ij")
    errs = torch.stack([ORE_g.flatten(), PLE_g.flatten()], dim=0)
    F_g  = _mc_fidelity(U_target, pulse, errs).reshape(1000, 50)

    return ORE_g.numpy(), PLE_g.numpy(), F_g.numpy(), F_mean, F_err


def _pulse_to_arrays(
    pulse: torch.Tensor,
    omega_max: float = OMEGA_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (phi, tau_us, Omega_x, Omega_y) arrays from a (L, 2) pulse tensor."""
    phi    = pulse[:, 0].numpy()
    tau_us = pulse[:, 1].numpy() / (2 * math.pi * omega_max)
    return phi, tau_us, np.cos(phi) * omega_max, np.sin(phi) * omega_max


def _smooth_arrays(
    phi: np.ndarray,
    tau_us: np.ndarray,
    cutoff_MHz: float,
    omega_max: float = OMEGA_MAX,
    oversample: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-domain LPF for a constant-amplitude pulse; returns (Ox_sm, Oy_sm)."""
    T     = tau_us.sum()
    dt    = tau_us.mean() / oversample
    M     = int(np.ceil(T / dt))
    edges = np.concatenate([[0.0], np.cumsum(tau_us)])
    phase = np.unwrap(phi)

    arr = np.zeros(M)
    for j in range(len(tau_us)):
        i0 = min(int(round(edges[j]     / dt)), M)
        i1 = min(int(round(edges[j + 1] / dt)), M)
        if i1 > i0:
            arr[i0:i1] = phase[j]

    nyq        = 0.5 / dt
    cutoff_eff = min(cutoff_MHz, nyq * 0.99)
    filt       = lowpass_filter(arr, dt, cutoff_eff)

    mids    = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.clip(np.round(mids / dt).astype(int), 0, len(filt) - 1)
    phase_sm = filt[idx]
    return omega_max * np.cos(phase_sm), omega_max * np.sin(phase_sm)


def _mc_fidelity_after_smooth(
    U_target: torch.Tensor,
    pulse: torch.Tensor,
    cutoff_MHz: float,
    M: int = MC_N,
) -> float:
    """MC-averaged fidelity (δ~N(0,1), ε~N(0,0.05²)) after LPF at cutoff_MHz."""
    phi, tau_us, _, _ = _pulse_to_arrays(pulse)
    ox_sm, oy_sm = _smooth_arrays(phi, tau_us, cutoff_MHz)
    phi_sm   = np.arctan2(oy_sm, ox_sm)
    pulse_sm = torch.tensor(
        np.stack([phi_sm, pulse[:, 1].numpy()], axis=1), dtype=torch.float32
    )
    err = ERROR_SAMPLERS["ore_ple"](M, delta_std=1.0, epsilon_std=0.05)
    return _mc_fidelity(U_target, pulse_sm, err).mean().item()


# ── Figure 1 – Fidelity contour (1 × 2, shared axes) ─────────────────────────
def fig_fidelity_contour() -> None:
    """1×2 fidelity contour: H gate (left) | X(π/2) gate (right)."""
    pipe = _load_pipe(
        "configs/transformer_len400.yaml",
        "demo_universal/weight/length_400.pt",
    )

    pulse_H, U_H = _get_pulse_and_target(pipe, _H_X, _H_Y, _H_Z, _H_TH)
    pulse_X, U_X = _get_pulse_and_target(pipe, _XH_X, _XH_Y, _XH_Z, _XH_TH)

    print("Computing contour for H gate …")
    ORE_H, PLE_H, F_H, Fm_H, Fe_H = _contour_data(U_H, pulse_H)
    print(f"  H:     E[F] = {Fm_H:.4f} ± {Fe_H:.4f}")

    print("Computing contour for X(π/2) gate …")
    ORE_X, PLE_X, F_X, Fm_X, Fe_X = _contour_data(U_X, pulse_X)
    print(f"  X(π/2): E[F] = {Fm_X:.4f} ± {Fe_X:.4f}")

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 5.5),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    cf = None
    for ax, ORE, PLE, F, label in zip(
        axes,
        [ORE_H, ORE_X],
        [PLE_H, PLE_X],
        [F_H,   F_X],
        ["(a)", "(b)"],
    ):
        cf = ax.contourf(ORE, PLE, F, levels=CONTOUR_LEVELS, cmap="viridis")
        ax.contour(ORE, PLE, F, levels=CONTOUR_WHITE, colors="white", linewidths=1.5)
        ax.grid(True, alpha=0.3)
        ax.text(0.0, 1.05, label, transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="right",
                clip_on=False)

    axes[0].set_ylabel(r"$\epsilon$")
    fig.supxlabel(r"$\delta / \Omega_{\mathrm{max}}$", fontsize=plt.rcParams["axes.labelsize"])

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Fidelity", fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "fig1_fidelity_contour.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 2 – Fidelity vs σ(δ/Ω_max) (1 × 2) ───────────────────────────────
def fig_fidelity_vs_std() -> None:
    """1×2 fidelity-vs-std: H gate for L=100 (left) | L=400 (right)."""
    try:
        import pwlf
    except ImportError as exc:
        raise ImportError("pip install pwlf") from exc

    pipe_100 = _load_pipe(
        "configs/transformer_len100.yaml",
        "demo_universal/weight/length_100.pt",
    )
    pipe_400 = _load_pipe(
        "configs/transformer_len400.yaml",
        "demo_universal/weight/length_400.pt",
    )

    pulse_100, U_H = _get_pulse_and_target(pipe_100, _H_X, _H_Y, _H_Z, _H_TH)
    pulse_400, _   = _get_pulse_and_target(pipe_400, _H_X, _H_Y, _H_Z, _H_TH)

    delta_list = list(np.arange(0.01, 2.0, 0.01))

    def sweep(U_target: torch.Tensor, pulse: torch.Tensor, tag: str) -> dict:
        out: dict = {}
        for ds in tqdm(delta_list, desc=f"fid-std {tag}"):
            err = ERROR_SAMPLERS["ore_ple"](MC_N, delta_std=float(ds), epsilon_std=0.05)
            F   = _mc_fidelity(U_target, pulse, err)
            out[float(ds)] = (F.mean().item(), F.std().item() / math.sqrt(MC_N))
        return out

    print("L=100 fidelity sweep …")
    fids_100 = sweep(U_H, pulse_100, "L=100")
    print("L=400 fidelity sweep …")
    fids_400 = sweep(U_H, pulse_400, "L=400")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    xl = r"$\mathrm{Std}(\delta / \Omega_{\mathrm{max}})$"

    for ax, fids, tag, label in zip(
        axes,
        [fids_100, fids_400],
        ["$L = 100$", "$L = 400$"],
        ["(a)", "(b)"],
    ):
        ds = list(fids.keys())
        Fm = [fids[d][0] for d in ds]
        Fe = [fids[d][1] for d in ds]

        m      = pwlf.PiecewiseLinFit(ds, Fm)
        breaks = m.fit(2)

        ax.scatter(ds, Fm, s=10, color="C0", label=tag, zorder=3)

        ax.set_ylim(0.8, 1.0)
        ax.grid(True)
        ax.text(0.0, 1.05, label, transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="right",
                clip_on=False)

    axes[0].set_ylabel("Expected Fidelity")
    fig.supxlabel(xl, fontsize=plt.rcParams["axes.labelsize"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "fig2_fidelity_vs_std.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 3 – Low-pass filter (2-column layout) ──────────────────────────────
def fig_lowpass(display_cutoff: float = 2000.0) -> None:
    """Left: Ωx, Ωy original vs filtered. Right: infidelity (log) vs cutoff freq."""
    pipe = _load_pipe(
        "configs/transformer_len400.yaml",
        "demo_universal/weight/length_400.pt",
    )
    pulse, U_H = _get_pulse_and_target(pipe, _H_X, _H_Y, _H_Z, _H_TH)

    err = ERROR_SAMPLERS["ore_ple"](MC_N, delta_std=1.0, epsilon_std=0.05)

    # fid_orig = _pure_fidelity(U_H, pulse)
    fid_orig = _mc_fidelity(U_H, pulse, err).mean().item()
    print(f"Original fidelity: {fid_orig:.6f}")
    print(f"Fidelity after {display_cutoff:.0f} MHz LPF: {_mc_fidelity_after_smooth(U_H, pulse, display_cutoff):.6f}")

    phi, tau_us, ox, oy = _pulse_to_arrays(pulse)
    t_edges_ns = np.concatenate([[0.0], np.cumsum(tau_us)]) * 1000.0  # μs → ns

    ox_sm, oy_sm = _smooth_arrays(phi, tau_us, display_cutoff)

    # Right panel: zero-noise infidelity vs cutoff frequency
    cutoffs = np.linspace(80, 8000, 100)
    print("Sweeping cutoff frequencies for infidelity curve …")
    infids = []
    for c in tqdm(cutoffs):
        F = _mc_fidelity_after_smooth(U_H, pulse, float(c))
        infids.append(max(1.0 - F, 1e-10))   # floor to avoid log(0)

    # Build 2-column layout: left column has 2 rows (Ωx, Ωy); right spans both
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
    ax_ox    = fig.add_subplot(gs[0, 0])
    ax_oy    = fig.add_subplot(gs[1, 0], sharex=ax_ox)
    ax_infid = fig.add_subplot(gs[:, 1])

    label_orig   = r"Original"
    label_filt   = rf"Filtered ($f_c = {display_cutoff:.0f}\ \mathrm{{MHz}}$)"

    # Ωx comparison
    ax_ox.step(t_edges_ns, np.r_[ox,    ox[-1]],    where="post",
               color="C0", label=label_orig)
    ax_ox.step(t_edges_ns, np.r_[ox_sm, ox_sm[-1]], where="post",
               color="C1", label=label_filt)
    ax_ox.set_ylabel(r"$\Omega_x\ (2\pi\ \mathrm{MHz})$")
    ax_ox.set_ylim(-1.1 * OMEGA_MAX, 1.1 * OMEGA_MAX)
    ax_ox.grid(True, alpha=0.4)
    ax_ox.legend(loc="upper right")
    plt.setp(ax_ox.get_xticklabels(), visible=False)

    # Ωy comparison
    ax_oy.step(t_edges_ns, np.r_[oy,    oy[-1]],    where="post", color="C0")
    ax_oy.step(t_edges_ns, np.r_[oy_sm, oy_sm[-1]], where="post", color="C1")
    ax_oy.set_ylabel(r"$\Omega_y\ (2\pi\ \mathrm{MHz})$")
    ax_oy.set_xlabel("Time (ns)")
    ax_oy.set_ylim(-1.1 * OMEGA_MAX, 1.1 * OMEGA_MAX)
    ax_oy.grid(True, alpha=0.4)

    # Infidelity vs cutoff
    ax_infid.plot(cutoffs, infids, "o-", ms=5, color="C2")
    # ax_infid.set_xscale("log")
    ax_infid.set_yscale("log")
    ax_infid.set_xlabel(r"Cutoff frequency $(2\pi\ \mathrm{MHz})$")
    ax_infid.set_ylabel(r"Infidelity $(1 - F)$")
    ax_infid.grid(True, which="both", alpha=0.4)
    ax_infid.axhline(1.0 - fid_orig, color="C0", linestyle="--", label="Original")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "fig3_lowpass.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate thesis figures.")
    p.add_argument("--figure", default="all",
                   choices=["1", "2", "3", "all"],
                   help="Which figure(s) to generate")
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.figure in ("1", "all"):
        fig_fidelity_contour()
    if args.figure in ("2", "all"):
        fig_fidelity_vs_std()
    if args.figure in ("3", "all"):
        fig_lowpass()
