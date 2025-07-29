"""
Reference:
https://arxiv.org/pdf/2312.08426
"""


from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd
import numpy as np
import pwlf

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors as mcolors

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import shutil

from qutip import Bloch


# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from train.single_qubit.single_qubit_script import *



#######################
# SCORE Dataset #######
#######################


angle_vec_dict = {
    1/4 : [1.34820, 1.32669, 1.77042, 2.16800],
    1/3 : [1.41901, 1.35864, 1.77664, 2.13759],
    1/2 : [1.55280, 1.42267, 1.78586, 2.07559],
    2/3 : [1.67478, 1.47865, 1.78919, 2.02043],
    3/4 : [1.73053, 1.49972, 1.78853, 1.99939],
    1   : [1.87342, 1.52524, 1.78436, 1.97330]
}

unitaries = {
    "X(pi)" : [(1, 0)],
    "X(pi-2)" : [(1/2, 0)],
    "Hadamard" : [(1, 0), (1/2, 1/2)],
    "Z(pi-4)" : [(1, 0), (1/2, 1/2), (1/4, 0), (1, 0), (1/2, 1/2)]
}


def SCOREn_config(n, phi):
    angle_vec = angle_vec_dict[n]
    config = []
    Angle = np.pi * n

    for i, angle in enumerate(angle_vec):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })
        Angle += (-1)**(len(angle_vec) - 1 - i) * 2 * angle * np.pi

    config.append({
        "phi": torch.tensor([phi], dtype=torch.complex64),
        "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
        "angle": torch.tensor([Angle], dtype=torch.complex64)
    })

    for i, angle in reversed(list(enumerate(angle_vec))):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })

    # config_to_tensor = [
    #     [0, 1, x["phi"], x["angle"]]
    #     for x in config
    # ]
    # config_to_tensor = [
    #     [x["phi"], x["angle"]]
    #     for x in config
    # ]

    dt = sum(x["angle"] for x in config) / 400

    config_to_tensor = []

    for x in config:
        phi = x["phi"].to(torch.float)
        angle = x["angle"].to(torch.float)
        N = math.ceil(angle / dt)
        for _ in range(N):
            config_to_tensor.append(
                [phi, angle / N]
            )

    return torch.tensor(config_to_tensor)


def build_SCORE_pulses(SCORE_emb=False):
    SCORE_pulses = []

    if SCORE_emb:
        unitaries = {
            angle : [(angle, 0)]
            for angle in angle_vec_dict
        }
    save_dir = "weights/SCORE_Pulse/"
    os.makedirs(save_dir, exist_ok=True)

    for target in unitaries:
        pulses = []

        for n, phi in reversed(unitaries[target]):
            pulses.append(SCOREn_config(n, phi * np.pi))
        
        SCORE_pulses.append(torch.stack(pulses))
        x = SCORE_pulses[-1]
        x = x.reshape(-1, x.shape[2]) 
        df = pd.DataFrame(x.to(torch.float))
        if type(target) is float:
            target = str(np.round(target, 2))
        df.to_csv(os.path.join(save_dir, f"{target}_SCORE_pulse.csv"), index=False)

    SCORE_pulses = [x.reshape(-1, x.shape[-1]) for x in SCORE_pulses]

    

    torch.save(SCORE_pulses, os.path.join(save_dir, "SCORE_pulse.pt"))

    return SCORE_pulses


#############################
# Pulse Parameter Plot ######
#############################


def plot_pulse_param(file_path, title, y_labels, df):
    # Extract pulse lengths (last column)
    x = df.iloc[:, len(y_labels)]

    # 1×2 grid: histogram (1 share) vs params (3 shares), with a bit more wspace
    fig, (ax_hist, ax_params) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(14, 6),
        gridspec_kw={'width_ratios': [1, 3], 'wspace': 0.4}
    )

    # --- Left: histogram ---
    ax_hist.hist(x / math.pi, bins=20, edgecolor='black')
    ax_hist.set_xlabel(r"Pulse Time (units of $\pi$)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Pulse Length Histogram")

    # --- Right: parameter stack ---
    if len(y_labels) == 1:
        axes = [ax_params]
    else:
        # remove the single placeholder and insert a vertical stack
        fig.delaxes(ax_params)
        axes = fig.add_gridspec(
            nrows=len(y_labels), ncols=1,
            left=0.40, right=0.98,  # you can also nudge these if needed
            top=0.90, bottom=0.10,
            hspace=0.3
        ).subplots()

    cumulative = np.concatenate(([0], np.cumsum(x / math.pi)))
    for i, ax in enumerate(axes):
        if i == len(axes) - 1:
            ax.step(cumulative[1:], df.iloc[:, i] / math.pi, where='post')
            ax.set_xlabel("Rotation time (units of π)")
        else:
            ax.step(cumulative[1:], df.iloc[:, i], where='post')
        ax.set_ylabel(y_labels[i])
        ax.grid(True)
        

    fig.suptitle(f"Composite Pulse for {title}", fontsize=16)

    os.makedirs(file_path, exist_ok=True)
    out_path = os.path.join(file_path, f"{title}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close(fig)


#############################
# Fideltiy Contour Plot #####
#############################


def fidelity_contour_plot(target_name, U_target, pulse, name, save_dir, M=10000, phase_only=True):

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    errors_mc = get_ore_ple_error_distribution(M, 1, 0.05)
    U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
    pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)

    g = batched_unitary_generator

    U_out_plot = g(pulses_plot, errors_mc)
    F = fidelity(U_out_plot, U_target_plot, 1)


    F_mean = F.mean().item()
    F_err = F.std().item() / np.sqrt(M)

    
    # Create 2D grid of ORE and PLE
    ORE_vals = torch.linspace(-3, 3, 1000)
    PLE_vals = torch.linspace(-0.15, 0.15, 50)
    ORE_grid, PLE_grid = torch.meshgrid(ORE_vals, PLE_vals, indexing="ij")

    # Flatten the grid for batch processing
    ORE_flat = ORE_grid.flatten()
    PLE_flat = PLE_grid.flatten()

    # Shape: (2, N)
    errors_grid = torch.stack([ORE_flat, PLE_flat], dim=0)

    # Repeat target and pulse
    N = errors_grid.shape[1]
    U_target_grid = U_target.expand(N, -1, -1)  # (N, d, d)
    pulses_grid = pulse.expand(N, -1, -1)       # (N, L, P)

    # Evaluate fidelity
    U_out_grid = g(pulses_grid, errors_grid)
    F_grid = fidelity(U_out_grid, U_target_grid, 1).reshape(1000, 50)

    # Convert meshgrid to numpy
    ORE_np = ORE_grid.detach().cpu().numpy()
    PLE_np = PLE_grid.detach().cpu().numpy()
    F_np = F_grid.numpy()

    plt.figure(figsize=(8, 6))
    # contour = plt.contourf(ORE_np, PLE_np, F_np, levels=20, cmap='viridis')
    contour = plt.contourf(ORE_np, PLE_np, F_np, levels=[0.8, 0.9, 0.95, 0.99, 0.999, 1.0], cmap='viridis')
    plt.contour(ORE_np, PLE_np, F_np, levels=[0.95, 0.99, 0.999], colors='white', linewidths=1.5)

    plt.colorbar(contour, label='Fidelity')
    plt.xlabel(r"$\delta / \Omega_{\max} \sim N(0, 1)$")
    plt.ylabel(r"$\epsilon / \Omega_{\max} \sim N(0, 0.05^2)$")
    plt.title(f"Fidelity Surface for {target_name} of {name}\nE[F] = {F_mean:.4f} +/- {F_err:.4f}\nTotal Evolution Time: {total_time:.2f} pi")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(
        os.path.join(save_dir, f"{target_name}.png")
    )



######################################
# Average Fidelity by std(delta) #####
######################################


def get_avg_fidelity(U_target, pulse, M=10000, phase_only=True, delta_list=None):

    fidelities = {}
    g = batched_unitary_generator

    if delta_list is None:
        delta_list = [0.1 * (i + 1) for i in range(10)]

    for delta_std in tqdm(delta_list):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)


        U_out_plot = g(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = f"{F_mean:.4f} +/- {F_err:.4f}"
    
    return fidelities


def plot_fidelity_by_std(target_name, U_target, pulse, name, save_dir, M=10000, phase_only=True):

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    fidelities = {}
    g = batched_unitary_generator

    delta_vals = torch.arange(0.01, 2.0, 0.01)

    for delta_std in tqdm(delta_vals):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)

        U_out_plot = g(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = (F_mean, F_err)

        
    # Extract values for plotting
    delta_vals = list(fidelities.keys())
    F_means = [fidelities[d][0] for d in delta_vals]
    inF_means = [1 - fidelities[d][0] for d in delta_vals]
    F_errs = [fidelities[d][1] for d in delta_vals]

    # Piece-wise Linear Fit
    num_segments = 2
    model = pwlf.PiecewiseLinFit(delta_vals, F_means)
    breaks = model.fit(num_segments)
    # Get slopes and intercepts
    slopes = model.slopes
    intercepts = model.intercepts
    F_pred = model.predict(delta_vals)

    # Plotting fidelity vs delta_std
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, F_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, F_pred, 'r--', label=f'Piecewise Linear Fit ({num_segments} segments)')

    # Annotate equations
    for i in range(num_segments):
        x_start = breaks[i]
        x_end = breaks[i+1]
        mid_x = (x_start + x_end) / 2
        mid_y = model.predict([mid_x])[0]
        
        eqn = f"y = {slopes[i]:.3f}x + {intercepts[i]:.3f}"
        plt.text(mid_x, mid_y - 0.03, eqn, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    plt.xlabel(r"Std$(\delta / \Omega_{\max})$")
    plt.ylabel("Expected Fidelity")
    plt.title(
        f"Fidelity curve for {target_name} of {name}\n"
        f"Total Evolution Time: {total_time:.2f} π"
    )
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(0.6, 1)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{target_name}_fidelity.png")
    )

    # Piece-wise Linear Fit
    num_segments = 3
    log_deltas = np.log(delta_vals)
    log_inF = np.log(inF_means)
    model = pwlf.PiecewiseLinFit(log_deltas, log_inF)
    breaks = model.fit(num_segments)
    # Get slopes and intercepts
    slopes = model.slopes
    intercepts = model.intercepts
    inF_pred = np.exp(model.predict(log_deltas))

    # Plotting log(infidelity) vs delta_std
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, inF_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, inF_pred, 'r--', label=f'Piecewise Linear Fit ({num_segments} segments)')

    plt.semilogy()
    plt.semilogx()
    # Annotate equations

    for i in range(num_segments):
        x_start = breaks[i]
        x_end = breaks[i+1]
        mid_x_log = (x_start + x_end) / 2
        mid_x = np.exp(mid_x_log)  # Transform back to delta_vals space
        mid_y = np.exp(model.predict([mid_x_log])[0])  # Also inverse-transform y

        eqn = f"log(y) = {slopes[i]:.3f} log(x) + {intercepts[i]:.3f}"
        plt.text(mid_x, mid_y * 1.2, eqn, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.xlabel(r"Std$(\delta / \Omega_{\max})$")
    plt.ylabel("Expected Infidelity")
    plt.title(
        f"Infidelity curve for {target_name} of {name}\n"
        f"Total Evolution Time: {total_time:.2f} π"
    )
    
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(1e-3, 1)
    plt.savefig(
        os.path.join(save_dir, f"{target_name}_infidelity_with_fit.png")
    )


######################################
# Qubit Ensemble Evolution Video #####
######################################

def animate_multi_error_bloch(
    bloch_vectors_list,   # list of arrays (T x 3)
    pulse_info_list,      # list of lists of pulse tuples
    fidelity_list,        # list of final fidelities
    delta_list,           # list of delta values
    epsilon_list,         # list of epsilon values
    name,
    save_path="multi_bloch_qutip.mp4",
    phase_only=True
):
    num_qubits = len(bloch_vectors_list)

    num_frames = bloch_vectors_list[0].shape[0]

    # Prepare colors and legend handles
    # colors = list(TABLEAU_COLORS.values())
    norm  = mcolors.Normalize(vmin=min(delta_list), vmax=max(delta_list))
    cmap  = cm.get_cmap('viridis')   # or any other built-in colormap
    colors = [cmap(norm(d)) for d in delta_list]
    
    # Create figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(name, fontsize=14)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(delta_list)   # any sequence of length ≥1 works
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(r'Detuning $\delta$')

    # Initialize Bloch object with our axes
    b = Bloch(fig=fig, axes=ax)
    b.view = [20, 45]
    # We'll override colors per trajectory, so set defaults to unused
    b.vector_color = colors
    b.point_color = colors

    # choose which index is τ (phase_only tuples are (something, φ, τ), otherwise (…, τ) at position 4)
    tau_idx = 2 if phase_only else 4

    # build an array: step_times[k] = total τ across all qubits at frame k
    step_times = []
    for k in range(num_frames):
        tot = 0.0
        for i in range(num_qubits):
            if k < len(pulse_info_list[i]):
                tot += pulse_info_list[i][k][tau_idx]
        step_times.append(tot / num_qubits)

    # cumulative time (in units of π)
    cumulative_times = np.cumsum(step_times) / np.pi

    def update(frame):
        # Clear previous frame
        b.clear()

        # Plot trajectories and vectors
        for i in range(num_qubits):
            traj = bloch_vectors_list[i][: frame + 1]
            xs, ys, zs = traj[:, 0].tolist(), traj[:, 1].tolist(), traj[:, 2].tolist()
            # trajectory line
            # b.add_points([xs, ys, zs], meth='l', colors=[to_rgba(colors[i % len(colors)]) for _ in range(3)], alpha=0.5)
            b.add_points([xs, ys, zs],
                         meth='l',
                         colors=[colors[i]]*len(xs),
                         alpha=0.5)
            # head of trajectory
            vec = bloch_vectors_list[i][frame]
            b.add_vectors([vec], colors=[colors[i]])

            # Hamiltonian arrow
            if pulse_info_list and frame < len(pulse_info_list[i]):
                if not phase_only:
                    _, D, O, phi, tau = pulse_info_list[i][frame]
                    ham_vec = [O * np.cos(phi), O * np.sin(phi), D]
                    b.add_vectors([ham_vec])
                else:
                    _, phi, tau = pulse_info_list[i][frame]
                    # ham_vec = [np.cos(phi), np.sin(phi), 0]
                
        T = cumulative_times[frame]
        F_mean = np.mean(fidelity_list)
        F_err = np.std(fidelity_list) / np.sqrt(len(fidelity_list))
        title_str = (
            f"{name}\n"
            fr"Total Time: {T:.4f}$\pi$"
            f"\nExpected Fidelity: {F_mean:.4f} +/- {F_err:.4f}"
        )
        fig.suptitle(title_str, fontsize=14)
        # Draw sphere and elements
        b.make_sphere()
        b.render()
        # Add legend
        # ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.05, 1.0), fontsize=8)

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    
    # ----------------------------------------------------------
    # choose a writer once
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:                              # MP4 via ffmpeg
        writer = FFMpegWriter(
            fps=num_frames // 10,                # set FPS *here*
            codec="libx264"
        )
    else:                                        # fallback: GIF
        save_path = os.path.splitext(save_path)[0] + ".gif"
        writer = PillowWriter(fps=num_frames // 10)

    # save –‑‑ NO fps / codec / bitrate / … here!
    ani.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    # ----------------------------------------------------------

