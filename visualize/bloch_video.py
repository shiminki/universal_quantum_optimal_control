"""Bloch-sphere ensemble-evolution animations (qutip.Bloch-backed)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from matplotlib.lines import Line2D


def spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    """Two-component complex spinor → (x, y, z) Bloch coordinates."""
    if psi.shape != (2,) or not torch.is_complex(psi):
        raise ValueError("psi must be a complex 2-vector")
    alpha, beta = psi[0], psi[1]
    x = 2 * torch.real(torch.conj(alpha) * beta)
    y = 2 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha) ** 2 - torch.abs(beta) ** 2
    return np.array([x.item(), y.item(), z.item()])


def animate_multi_error_bloch(
    bloch_vectors_list: List[np.ndarray],
    pulse_info_list: List[List[tuple]],
    fidelity_list: List[float],
    delta_list: List[float],
    epsilon_list: List[float],
    name: str,
    save_path: str = "multi_bloch_qutip.mp4",
    phase_only: bool = True,
    Omega: Optional[float] = None,
) -> None:
    """Render an ensemble of Bloch trajectories under different (δ, ε)."""
    try:
        from qutip import Bloch
    except ImportError as e:
        raise ImportError("animate_multi_error_bloch needs `pip install uqoc[viz]` (qutip)") from e

    num_qubits = len(bloch_vectors_list)
    num_frames = bloch_vectors_list[0].shape[0]
    colors = list(TABLEAU_COLORS.values())[:num_qubits + 1]

    legend_handles = [
        Line2D([0], [0], color=colors[i % 10], lw=2,
               label=fr"$\delta$={delta_list[i % len(delta_list)]:.2f}, F={fidelity_list[i]:.4f}")
        for i in range(num_qubits)
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(name, fontsize=14)
    b = Bloch(fig=fig, axes=ax)
    b.view = [20, 45]
    b.vector_color = colors
    b.point_color = colors

    tau_idx = 2 if phase_only else 4
    step_times = [
        np.mean([pulse_info_list[i][k][tau_idx]
                 for i in range(num_qubits) if k < len(pulse_info_list[i])])
        if any(k < len(pulse_info_list[i]) for i in range(num_qubits)) else 0.0
        for k in range(num_frames)
    ]
    cumulative_times = np.cumsum(step_times) / np.pi

    def update(frame: int) -> None:
        b.clear()
        for i in range(num_qubits):
            traj = bloch_vectors_list[i][: frame + 1]
            xs, ys, zs = traj[:, 0].tolist(), traj[:, 1].tolist(), traj[:, 2].tolist()
            b.add_points([xs, ys, zs], meth='l',
                         colors=[to_rgba(colors[i % 10]) for _ in range(3)], alpha=0.5)
            b.add_vectors([bloch_vectors_list[i][frame].tolist()], colors=colors[i % 10])
            if pulse_info_list and frame < len(pulse_info_list[i]) and not phase_only:
                _, D, O, phi, _ = pulse_info_list[i][frame]
                b.add_vectors([[O * np.cos(phi), O * np.sin(phi), D]])

        T = cumulative_times[frame]
        if Omega is not None:
            title_str = (f"{name}\n"
                         fr"Total Time: {(1000 * T / (2 * Omega)):.4f} ns ({T:.4f}$\pi)$" + "\n"
                         f"E[F] = {np.mean(fidelity_list):.4f} +/- "
                         f"{np.std(fidelity_list) / np.sqrt(len(fidelity_list)):.4f}")
        else:
            title_str = (f"{name}\n"
                         fr"Total Time: {T:.4f}$\pi$" + "\n"
                         f"E[F] = {np.mean(fidelity_list):.4f} +/- "
                         f"{np.std(fidelity_list) / np.sqrt(len(fidelity_list)):.4f}")
        fig.suptitle(title_str, fontsize=14)
        b.make_sphere(); b.render()
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.05, 1.0), fontsize=8)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    ani.save(save_path, fps=15, dpi=150)
    plt.close(fig)
