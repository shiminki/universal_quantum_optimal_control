from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model_encoder import CompositePulseTransformerEncoder
from trainer import CompositePulseTrainer

from run.single_qubit.single_qubit_script import *



def single_fidelity(A, B):
    trace = torch.trace(A.conj().T @ B)
    return (trace.abs() ** 2 + 2) / 6


if __name__ == "__main__":
    torch.manual_seed(0)
    M = 10000

    # pulse_path = "weights/100_pulses/_err_{'delta_std':tensor(1.)}_pulses.pt"
    # pulse_path = "weights/_err_{_delta_std_tensor(1.),_epsilon_std_0.05}_pulses.pt"
    pulse_path = "_err_{'delta_std':tensor(1.)}_pulses.pt"
    is_old = True

    pulses = torch.load(pulse_path) # [4, 100, 4]
    # pulses_mc = pulses.repeat_interleave(M, dim=0)

    train_set = build_dataset() # [4, 2, 2]

    train_set_name = [
        "X(pi)", "X(pi-2)", "Hadamard", "Z(pi-4)"
    ]

    for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
        df = pd.DataFrame(pulse)
        df.to_csv(f"pulses/{target_name}_pulse.csv", index=False)

        if is_old:
            df[3] *= 2

        errors_mc = get_ore_ple_error_distribution(M, 1, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)


        U_out_plot = batched_unitary_generator(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        
        # Create 2D grid of ORE and PLE
        ORE_vals = torch.linspace(-3, 3, 1000)
        PLE_vals = torch.linspace(-0.3, 0.3, 50)
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
        contour = plt.contourf(ORE_np, PLE_np, F_np, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Fidelity')
        plt.xlabel(r"ORE ({\delta/\Omega_\max})")
        plt.ylabel(r"PLE ({\epsilon/\Omega_\max})")
        plt.title(f"Fidelity Surface for {target_name}\nF = {F.mean():.4f}")
        plt.grid(True)
        plt.show()

        