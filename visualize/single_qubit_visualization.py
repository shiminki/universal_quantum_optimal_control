from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm


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
    plt.title(f"Fidelity Surface for {target_name} of {name}\nE[F] = {F_mean:.4f} +/- {F_err:.4f}\nTotal Evolution Time: {total_time:.2f} pi")
    plt.grid(True)
    plt.show()



def get_avg_fidelity(U_target, pulse):

    fidelities = {}

    for delta_std in tqdm((0.1 * (i + 1) for i in range(10))):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)


        U_out_plot = batched_unitary_generator(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = f"{F_mean:.4f} +/- {F_err:.4f}"
    
    return fidelities
        


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

    fidelities = {}

    # for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
    #     fidelities[target_name] = get_avg_fidelity(U_target, pulse)
    
    # df = pd.DataFrame(fidelities)
    # df.to_csv("fidelities.csv")
    

    # for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
    #     visualize(target_name, U_target, pulse, "Transformer CP", True)



    SCORE_pulses = build_SCORE_pulses()

    for target_name, U_target, pulse in zip(train_set_name, train_set, SCORE_pulses):

        print(pulse)

        print(pulse.shape)
        visualize(target_name, U_target, pulse, "SCORE4")


    data = [
        [0.7598, 1.1145, -0.8872, -0.0167, -1.5056, 0.4233, -0.0248, 2.9724, -3.0918],
        [1.1516, 0.3675, -2.2800, 0.5743, 2.0933, 0.2061, -1.6437, -0.7061, 0.9273],
        [0.6220, 0.6251, 1.1929, 1.6412, -0.8985, -1.5013, -0.5969, -0.6176, 1.2448],
        [2.7879, 1.8734, -2.8629, 1.2522, 2.4905, -1.1821, -1.8967, -2.1757, 2.4733],
        [-2.8126, -2.7701, -1.0354, -2.8256, 2.3944, -1.7776, -0.7125, 0.5963, -1.4869],
        [-1.2035, 1.7050, 2.0577, -0.7816, 0.7853, -1.3623, 0.9122, 0.0378, 2.7522]
    ]

    for i, phis in enumerate(data):
        
        Deltas = [0] * len(phis)
        Omegas = [1] * len(phis)
        taus = [torch.pi] * len(phis)

        pulse = torch.tensor([
            Deltas, Omegas, phis, taus
        ]).T

        U_target = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.cfloat)

        visualize("X(pi)", U_target, pulse, f"NN_{i}")
