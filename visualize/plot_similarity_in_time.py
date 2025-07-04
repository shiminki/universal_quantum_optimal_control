
import pandas as pd
from qutip import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import sys
import os
from tqdm import tqdm

# Add parent directory (adjust the number of '..' as needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from train.single_qubit.single_qubit_script import *


_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]


def similarity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Trace sequed"""
    # trace over last two dims, keep batch
    # Batched conjugate transpose and matrix multiplication
    U_out_dagger = U_out.conj().transpose(-1, -2)  # [batch, 2, 2]
    product = U_out_dagger @ U_target  # [batch, 2, 2]

    # print(product, product.shape)

    # Batched trace calculation
    trace = torch.einsum('bii->b', product)  # [batch]
    trace_squared = torch.abs(trace) ** 2

    d = 2 ** num_qubits

    return (trace_squared) / (d **2)


def get_similarity_by_time(pulse):

    mean_similarities = []
    err_similarities = []
    M = 10000

    T = pulse.shape[0]
    for t in tqdm(range(1, T + 1)):
        errors_mc = get_ore_ple_error_distribution(2 * M, 1, 0.05)
        sub_pulse = pulse[:t, :]
        pulses_plot = torch.stack([sub_pulse]).repeat_interleave(2 * M, dim=0)

        U_out_plot = batched_unitary_generator(pulses_plot, errors_mc)
        U_out_plot1 = U_out_plot[:M]
        U_out_plot2 = U_out_plot[M:]
        F = similarity(U_out_plot1, U_out_plot2, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        mean_similarities.append(F_mean)
        err_similarities.append(F_err)
    
    return mean_similarities, err_similarities


if __name__ == "__main__":
    torch.manual_seed(0)
    M = 10000

    SCORE_embedding = True

    # pulse_path = "weights/single_qubit_control/err_{_delta_std_tensor(1.),_epsilon_std_0.05}_pulses.pt"
    pulse_path = "weights/single_qubit_control/SCORE Embedding/err_{_delta_std_tensor(1.3000),_epsilon_std_0.05}_pulses.pt"
    # pulse_path = "weights/single_qubit_control/err_{_delta_std_tensor(1.6000),_epsilon_std_0.05}_pulses.pt"
    
    # pulse_path = "weights/single_qubit_control/no_SCORE_err_{_delta_std_tensor(1.3000),_epsilon_std_0.05}_pulses.pt"
    pulses = torch.load(pulse_path) # [4, 100, 4]
    # pulses_mc = pulses.repeat_interleave(M, dim=0)

    # print(pulses.shape)

    
    if SCORE_embedding:
        _, train_set = build_score_emb_dataset()

        train_set_name = [
            fr"$R_X$({n:.2f}$\pi$)"
            for n in (1/4, 1/3, 1/2, 2/3, 3/4, 1)
        ]
    else:
        train_set = build_dataset() # [4, 2, 2]

        train_set_name = [
            "X(pi)", "X(pi-2)", "Hadamard", "Z(pi-4)"
        ]

    for target_name, U_target, pulse in zip(train_set_name, train_set, pulses):
        time = pulse[:, 3]
        total_time = [0]
        for t in time:
            total_time.append(total_time[-1] + t / math.pi)

        mean_similarity, err_similarlity = get_similarity_by_time(pulse)
        plt.errorbar(total_time[1:], mean_similarity, err_similarlity)
        plt.xlabel(r"Time (units of $\pi$)")
        plt.ylabel("Expected Similarity")
        plt.title(fr"{target_name} Pairwise Similarity by time")
        plt.show()

