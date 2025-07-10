# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from util import *

from train.single_qubit.single_qubit_script import *
from train.single_qubit_phase_only.single_qubit_phase_control import batched_unitary_generator as batched_unitary_phase_control


#######################################################
# Qubit Ensemble Evolution Video Helper Functions #####
#######################################################


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


def generate_unitary_phase_only(pulse, delta, epsilon):
    phi = pulse[1]
    tau = pulse[2] / 2
    H_base = (np.cos(phi) * pauli[1] + np.sin(phi) * pauli[2])
    H = (H_base + delta * pauli[3])
    return torch.linalg.matrix_exp(-1j * H * tau * (1 + epsilon))


# Convert spinor to Bloch vector
def spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    assert psi.shape == (2,), "psi must be a 2D complex vector"
    assert torch.is_complex(psi), "psi must be complex-valued"
    alpha, beta = psi[0], psi[1]
    x = 2 * torch.real(torch.conj(alpha) * beta)
    y = 2 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha)**2 - torch.abs(beta)**2
    return np.array([x.item(), y.item(), z.item()])


###################
# Driver Code #####
###################


if __name__ == "__main__":
    # model_name = "Transformer_Phase_Control_Only"
    # phase_control_only = True
    # pulse_dir = "weights/phase_control/err_{'delta_std':tensor(1.3000),'epsilon_std':0.05}_pulses.pt"
    # save_dir = "figures/phase_control_only"


    # model_name = "Transformer_old_CP"
    # phase_control_only = False
    # pulse_dir = "Old Files/weights/single_qubit_control/SCORE Embedding/err_{_delta_std_tensor(1.),_epsilon_std_0.05}_pulses.pt"
    # save_dir = "figures/old_CP"

    # tau_max = "0.07"

    # model_name = f"Transformer_Phase_Control_{tau_max}_tau_max"
    # phase_control_only = True
    # pulse_dir = (
    #     f"weights/phase_control_{tau_max}_tau_max/"
    #     "err_{_delta_std_tensor(0.7000),_epsilon_std_0.05}_pulses.pt"
    #     # "err_{'delta_std':tensor(1.),'epsilon_std':0.05}_pulses.pt"
    # )
    # save_dir = f"figures/phase_control_{tau_max}_tau_max"

    # model_name = "Transformer_output_postprocessed"
    # phase_control_only = True
    # pulse_dir = "weights/fine_tuned_pulse/err_{_delta_std_tensor(1.),_epsilon_std_0.05}_pulses.pt"
    # save_dir = "figures/finetuned_pulse"

    # SCORE_embedding = True

    model_name = "SCORE4 Pulse"
    phase_control_only = True
    pulse_dir = "weights/SCORE_Pulse/SCORE_pulse.pt"
    save_dir = "figures/SCORE4"

    SCORE_embedding = True


    pulses = torch.load(pulse_dir)
    
    os.makedirs(save_dir, exist_ok=True)

    PSI_INIT = torch.tensor([1, 0], dtype=torch.cfloat)

    # y_labels = [
    #     "Detuning / max Rabi",
    #     "Rabi Frequency",
    #     "Phase (units of pi)"
    # ]

    y_labels = [
        "Phase (units of pi)"
    ]


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
        print(f"Figures for {target_name}")


        # Load and save plot param csv
        df = pd.DataFrame(pulse)
        csv_dir = os.path.join(save_dir, "pulse_param_csv")
        os.makedirs(csv_dir, exist_ok=True)
        pulse_csv_dir = os.path.join(csv_dir, f"{target_name}_pulse.csv")
        df.to_csv(pulse_csv_dir, index=False)

        # Plot fidelity contour plot
        print("Generating fidelity contour plot")
        fidelity_contour_plot(
            target_name, U_target, pulse, model_name, 
            os.path.join(save_dir, "fidelity_contour_plot"),
            phase_only=phase_control_only
        )

        # Plot the params as a function of time
        print("Generating pulse param plot")

        plot_pulse_param(
            os.path.join(save_dir, "pulse_param"), 
            target_name, y_labels, df
        )

        
        # Plot fidelity vs std(delta)
        print("Generating fidelity vs delta_std plot")
        plot_fidelity_by_std(
            target_name, U_target, pulse, model_name,
            os.path.join(save_dir, "fidelity_vs_delta_std"),
            phase_only=phase_control_only
        )


        # Generate qubit evolution video
        print("Generating qubit evolution video")
        deltas = [-2, -1, -0.5, 0, 0.5, 1, 2]
        epsilons = [0]
        bloch_list, pulse_info_list, fidelity_list = [], [], []

        # target state
        target_psi = U_target @ PSI_INIT

        for eps in epsilons:
            for delt in deltas:
                # simulate
                psi = PSI_INIT
                bv, pi = [], []
                # tau = 0
                for p in df.itertuples():
                    g = generate_unitary if not phase_control_only else generate_unitary_phase_only
                    U = g(p, delta=delt, epsilon=eps)
                    psi = U @ psi
                    bv.append(spinor_to_bloch(psi))
                    if phase_control_only:
                        # tau += p[2]
                        pi.append((0, p[1], p[2]))
                    else:
                        # tau += p[4]
                        pi.append((0, p[1], p[2], p[3], p[4]))
  
                bloch_list.append(np.vstack(([spinor_to_bloch(PSI_INIT)], bv)))
                pulse_info_list.append(pi)
                fidelity_list.append(np.abs(torch.vdot(target_psi, psi))**2)
        
        video_dir = os.path.join(save_dir, "qubit_evolutions")
        os.makedirs(video_dir, exist_ok=True)
        animate_multi_error_bloch(
            bloch_list, pulse_info_list, fidelity_list,
            deltas, epsilons,
            name=f"Ensemble Evolution of {target_name}",
            save_path=os.path.join(video_dir, f"{target_name}.mp4"),
            phase_only=phase_control_only
        )