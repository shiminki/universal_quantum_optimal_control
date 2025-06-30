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


def plot_fidelity_by_std(target_name, U_target, pulse, name):

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    fidelities = {}

    delta_vals = torch.arange(0, 1.0, 0.01)

    for delta_std in tqdm(delta_vals):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)

        U_out_plot = batched_unitary_generator(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = (F_mean, F_err)
        
    # Extract values for plotting
    delta_vals = list(fidelities.keys())
    F_means = [1 - fidelities[d][0] for d in delta_vals]
    F_errs = [fidelities[d][1] for d in delta_vals]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, F_means, yerr=F_errs, fmt='o-', capsize=4)

    plt.xlabel(r"Std$(\delta / \Omega_{\max})$")
    plt.ylabel("Expected Infidelity")
    plt.title(
        f"Infidelity curve for {target_name} of {name}\n"
        f"Total Evolution Time: {total_time:.2f} π"
    )
    plt.semilogy()
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(1e-3, 1)
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

    

    for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
        plot_fidelity_by_std(target_name, U_target, pulse, "Transformer CP")



    SCORE_pulses = build_SCORE_pulses()

    for target_name, U_target, pulse in zip(train_set_name, train_set, SCORE_pulses):
        plot_fidelity_by_std(target_name, U_target, pulse, "SCORE4")

