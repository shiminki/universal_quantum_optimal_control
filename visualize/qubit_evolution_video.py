
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



def spinor_to_bloch(psi: torch.Tensor) -> torch.Tensor:
    assert psi.shape == (2,), "psi must be a 2D complex vector"
    assert torch.is_complex(psi), "psi must be complex-valued"
    alpha, beta = psi[0], psi[1]
    x = 2 * torch.real(torch.conj(alpha) * beta)
    y = 2 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha)**2 - torch.abs(beta)**2
    return np.array([x, y, z])


def set_axes_equal_custom(ax, z_range=5):
    """Make axes of 3D plot have equal scale and size, centered around z-range."""
    radius = z_range
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-z_range, z_range])



def animate_bloch_evolution(bloch_vectors, name, pulse_info, delta, epsilon, save_path="qubit_evolution.mp4"):
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(10, 1)
    ax_text = fig.add_subplot(gs[0, 0])   # top 1/10 (10%)
    ax_bloch = fig.add_subplot(gs[1:, 0], projection='3d')  # bottom 9/10 (90%)

    ax_text.axis("off")  # hide axis
    text_obj = ax_text.text(0.5, 0.5, "", ha='center', va='center', fontsize=12, wrap=True)


    b = Bloch(fig=fig, axes=ax_bloch)
    b.vector_color = ['r']
    b.point_color = ['b']
    b.point_marker = ['o']
    b.point_size = [30]

    def update(i):
        b.clear()

        # Line connecting previous trajectory
        xline, yline, zline = bloch_vectors[:i+1].T
        b.add_points([xline, yline, zline], meth='l')  # connect dots with line

        # Add points
        b.add_points([xline, yline, zline], meth='s')  # also show spheres at steps

        # Bloch state vector
        b.add_vectors([bloch_vectors[i]])

        # Hamiltonian vector
        if i < len(pulse_info):
            tau_sum, Delta, Omega, phi = pulse_info[i]
            text_obj.set_text(
                f"Target Unitary = {name}\nORE = {delta}, PLE = {epsilon}\nCumulative time: {(tau_sum / np.pi):.2f} pi\nΔ = {Delta:.2f}, Ω = {Omega:.2f}, φ = {(phi / np.pi):.2f} pi"
            )
            hamiltonian = (Omega * np.cos(phi), Omega * np.sin(phi), Delta)
            b.add_vectors([hamiltonian])

        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Qubit State'),
            Line2D([0], [0], color='orange', lw=2, label='Control Hamiltonian')
        ]
        ax_bloch.legend(handles=legend_elements, loc='upper left', fontsize=10)

        b.vector_color = ['blue', 'orange']
        b.make_sphere()

        # set_axes_equal_custom(b.axes, z_range=5)  # expand all axes equally to match vertical




    ani = FuncAnimation(fig, update, frames=len(bloch_vectors), repeat=False)
    ani.save(save_path, fps=40, dpi=200)
    print(f"Video saved to {save_path}")


def analyze(pulse_dir, U_target, name, delta, epsilon):
    pulses = pd.read_csv(pulse_dir)

    _I2_CPU = torch.eye(2, dtype=torch.cfloat)
    _SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
    _SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
    _SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
    pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]

    def generate_unitary(pulse, delta, epsilon):
        Delta = pulse[1]
        Omega = pulse[2]
        phi = pulse[3]
        tau = pulse[4] / 2
        H_base = (Delta * pauli[3] +
                  Omega * (np.cos(phi) * pauli[1] + np.sin(phi) * pauli[2]))
        H = (H_base + delta * pauli[3])
        return torch.linalg.matrix_exp(-1j * H * tau * (1 + epsilon))

    # psi_init = torch.tensor([1, 0], dtype=torch.cfloat)
    psi_init = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.cfloat)
    bloch_3d_vec = [spinor_to_bloch(psi_init)]
    U_out = _I2_CPU
    evolution_time = 0


    pulse_info = []
    tau_cumsum = 0

    for pulse in pulses.itertuples():
        U_step = generate_unitary(pulse, delta=0.5, epsilon=0.0)
        psi_init = U_step @ psi_init
        U_out = U_step @ U_out
        bloch_3d_vec.append(spinor_to_bloch(psi_init))
        tau_cumsum += pulse[4]
        evolution_time += pulse[4]
        pulse_info.append((tau_cumsum, pulse[1], pulse[2], pulse[3]))  # Delta, Omega, phi


    bloch_3d_vec = np.array(bloch_3d_vec)
    animate_bloch_evolution(bloch_3d_vec, name, pulse_info, delta, epsilon, f"qubit_evolution/{name}_bloch_delta_{delta}_epsilon_{epsilon}.mp4")
    print(U_out)
    print(f"Total Evolution time: {evolution_time / np.pi} pi")

# Example usage
if __name__ == "__main__":
    from train.single_qubit.single_qubit_script import build_dataset
    U_targets = build_dataset()
    U_target_names = ["X(pi)", "X(pi-2)", "Hadamard", "Z(pi-4)"]
    for U_target, name in zip(U_targets, U_target_names):
        for delta in (0, 0.5, -0.5):
            pulse_dir = f"weights/single_qubit_control/{name}_pulse.csv"
            print(name)
            analyze(pulse_dir, U_target, name, delta, 0)
