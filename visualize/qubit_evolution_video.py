
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




def animate_bloch_grid_evolution(
    bloch_vectors_list, target_psi, name, pulse_info_list, delta_list, epsilon_list, save_path="bloch_grid_with_paths.mp4"
):
    assert len(bloch_vectors_list) == 15
    num_qubits = 15
    num_frames = len(bloch_vectors_list[0])

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(name, fontsize=16)
    axes = []
    arrows = []
    hams = []
    paths = []

    for i in range(3):
        for j in range(5):
            idx = 5 * i + j
            ax = fig.add_subplot(3, 5, idx + 1, projection='3d')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"Δ={delta_list[j]}, ε={epsilon_list[i]}", fontsize=12)
            ax.view_init(elev=20, azim=45)
            axes.append(ax)

            # Initial qubit vector
            vec = bloch_vectors_list[idx][0]
            arrow = ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='blue')
            arrows.append(arrow)

            # Initial Hamiltonian vector
            if pulse_info_list:
                _, D, O, phi = pulse_info_list[idx][0]
                hx, hy, hz = O * np.cos(phi), O * np.sin(phi), D
                ham = ax.quiver(0, 0, 0, hx, hy, hz, color='orange')
                hams.append(ham)
            else:
                hams.append(None)

            # Trajectory path (start with one point)
            x0, y0, z0 = vec
            path_line, = ax.plot([x0], [y0], [z0], color='gray', linewidth=1)
            paths.append(path_line)

            def update(frame):
                fidelity = {}
                for idx in range(num_qubits):
                    vecs = bloch_vectors_list[idx]
                    vec = vecs[frame]
                    x, y, z = vec

                    # Update Bloch vector
                    arrows[idx].remove()
                    arrows[idx] = axes[idx].quiver(0, 0, 0, x, y, z, color='blue')

                    # Update Hamiltonian
                    if pulse_info_list and frame < len(pulse_info_list[idx]):
                        _, D, O, phi = pulse_info_list[idx][frame]
                        hx, hy, hz = O * np.cos(phi), O * np.sin(phi), D
                        if hams[idx]: hams[idx].remove()
                        hams[idx] = axes[idx].quiver(0, 0, 0, hx, hy, hz, color='orange')

                    # Update trajectory path
                    path_line = paths[idx]
                    old_x, old_y, old_z = path_line._verts3d
                    new_x = np.append(old_x, x)
                    new_y = np.append(old_y, y)
                    new_z = np.append(old_z, z)
                    path_line.set_data_3d(new_x, new_y, new_z)

                    # Compute fidelity = |<target|psi>|^2
                    # Assume current state is |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
                    theta = np.arccos(z)
                    phi = np.arctan2(y, x)
                    current_psi = np.array([
                        np.cos(theta / 2),
                        np.exp(1j * phi) * np.sin(theta / 2)
                    ])


                    fidelity[idx] = np.abs(np.vdot(target_psi, current_psi))**2

                    # Update title with Δ, ε, vec, and fidelity
                    i, j = divmod(idx, 5)
                    delta = delta_list[j]
                    epsilon = epsilon_list[i]
                    axes[idx].set_title(
                        f"Δ={delta}, ε={epsilon}\n"
                        f"(x,y,z)=({x:.2f},{y:.2f},{z:.2f})\n"
                        f"|⟨ψ_target|ψ⟩|² = {fidelity[idx]:.4f}",
                        fontsize=9
                    )

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)
    ani.save(save_path, fps=30, dpi=150)
    plt.close()



def analyze(pulse_dir, U_target, name):
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

    
    PSI_INIT = torch.tensor([1, 0], dtype=torch.cfloat)

    def get_bloch_vec_and_pulse_info(delta, epsilon):
        psi_init = PSI_INIT
        # psi_init = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.cfloat)
        bloch_3d_vec = [spinor_to_bloch(psi_init)]
        U_out = _I2_CPU
        evolution_time = 0

        pulse_info = []
        tau_cumsum = 0

        for pulse in pulses.itertuples():
            U_step = generate_unitary(pulse, delta=delta, epsilon=epsilon)
            psi_init = U_step @ psi_init
            U_out = U_step @ U_out
            bloch_3d_vec.append(spinor_to_bloch(psi_init))
            tau_cumsum += pulse[4]
            evolution_time += pulse[4]
            pulse_info.append((tau_cumsum, pulse[1], pulse[2], pulse[3]))  # Delta, Omega, phi

        bloch_3d_vec = np.array(bloch_3d_vec)
        return bloch_3d_vec, pulse_info

    bloch_3d_vec_list = []
    pulse_info_list = []
    delta_list = [-2, -1, 0, 1, 2]
    epsilon_list = [-0.1, 0, 0.1]

    for epsilon in epsilon_list:
        for delta in delta_list:
            bloch_3d_vec, pulse_info = get_bloch_vec_and_pulse_info(delta, epsilon)
            bloch_3d_vec_list.append(bloch_3d_vec)
            pulse_info_list.append(pulse_info)

    target_psi = U_target @ PSI_INIT

    print(target_psi)

    animate_bloch_grid_evolution(bloch_3d_vec_list, target_psi,  name, pulse_info_list, delta_list, epsilon_list, f"qubit_evolution/{name}_bloch.mp4")
    

# Example usage
if __name__ == "__main__":
    from train.single_qubit.single_qubit_script import build_dataset, build_score_emb_dataset
    SCORE_embedding = True

    if SCORE_embedding:
        _, U_targets = build_score_emb_dataset()
        U_target_names = ["0-25", "0-33", "0-50", "0-67", "0-75", "1-00"]
    else:
        U_targets = build_dataset()
        U_target_names = ["X(pi)", "X(pi-2)", "Hadamard", "Z(pi-4)"]  


    for U_target, name in zip(U_targets, U_target_names):
        if SCORE_embedding:
            decimal = name.split('-')[1]
            digit = "1" if decimal == "00" else "0"
            title = fr"{digit}.{decimal}$\pi$"
        else:
            title = name
        
        pulse_dir = f"weights/single_qubit_control/SCORE Embedding/{name}_pulse.csv"
        print(name)

        analyze(pulse_dir, U_target, title)
