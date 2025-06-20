from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from train.single_qubit.single_qubit_script import *
from visualize.SCORE_visualization import *


def single_fidelity(A, B):
    trace = torch.trace(A.conj().T @ B)
    return (trace.abs() ** 2 + 2) / 6


def visualize(target_name, U_target, pulse, name, save_csv=False):

    if save_csv:
        df = pd.DataFrame(pulse)
        df.to_csv(f"weights/single_qubit_control/{target_name}_pulse.csv", index=False)

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    errors_mc = get_ore_ple_error_distribution(M, 1, 0.05)
    U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
    pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)


    U_out_plot = batched_unitary_generator(pulses_plot, errors_mc)
    F = fidelity(U_out_plot, U_target_plot, 1)

    
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
    U_out_grid = batched_unitary_generator(pulses_grid, errors_grid)
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
    plt.title(f"Fidelity Surface for {target_name} of {name}\nE[F] = {F.mean().item():.4f}\nTotal Evolution Time: {total_time:.2f} pi")
    # plt.suptitle(f"Fidelity Surface for {target_name}")
    # plt.title(
    #     fr"$\mathbb{{E}}_{{\sigma_\delta = 1,\ \sigma_\epsilon = 0.05}}[F] = {F.mean().item():.4f}$",
    #     # fontsize=18
    # )
    plt.grid(True)
    plt.show()

        


if __name__ == "__main__":
    torch.manual_seed(0)
    M = 10000

    pulse_path = "weights/single_qubit_control/_err_{_delta_std_tensor(1.),_epsilon_std_0.05}_pulses.pt"
    

    pulses = torch.load(pulse_path) # [4, 100, 4]
    # pulses_mc = pulses.repeat_interleave(M, dim=0)

    train_set = build_dataset() # [4, 2, 2]

    train_set_name = [
        "X(pi)", "X(pi-2)", "Hadamard", "Z(pi-4)"
    ]

    for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
        visualize(target_name, U_target, pulse, "Transformer CP", True)

    SCORE_pulses = build_SCORE_pulses()

    for target_name, U_target, pulse in zip(train_set_name, train_set, SCORE_pulses):

        print(pulse)

        print(pulse.shape)
        visualize(target_name, U_target, pulse, "SCORE4")


