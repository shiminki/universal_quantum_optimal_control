# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from util import *


#######################################################
# Qubit Ensemble Evolution Video Helper Functions #####
#######################################################


_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]


# Convert spinor to Bloch vector
def spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    assert psi.shape == (2,), "psi must be a 2D complex vector"
    assert torch.is_complex(psi), "psi must be complex-valued"
    alpha, beta = psi[0], psi[1]
    x = 2 * torch.real(torch.conj(alpha) * beta)
    y = 2 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha)**2 - torch.abs(beta)**2
    return np.array([x.item(), y.item(), z.item()])


def generate_unitary(pulse, delta, epsilon):
    phi = pulse[1]
    tau = pulse[2] / 2
    H_base = (np.cos(phi) * pauli[1] + np.sin(phi) * pauli[2])
    H = (H_base + delta * pauli[3])
    return torch.linalg.matrix_exp(-1j * H * tau * (1 + epsilon))


def rotation_to_unitary(rotation_vector):
    """
    Convert a rotation vector to a unitary matrix.
    The rotation vector is expected to be in the form [x, y, z, theta].
    """
    x, y, z, theta = rotation_vector
    n = np.sqrt(x**2 + y**2 + z**2)
    if n == 0:
        return torch.eye(2, dtype=torch.cfloat)
    
    axis = np.array([x/n, y/n, z/n])
    H = 0.5 * theta * (axis[0]*_SIGMA_X_CPU + axis[1]*_SIGMA_Y_CPU + axis[2]*_SIGMA_Z_CPU)
    return torch.matrix_exp(-1j * H)


from model.universal_model import UniversalQOCTransformer, Pipeline
from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import load_model_params

weight_path = "weights/universal_gates/universal_score_len100_pretrain/err_{_delta_std_tensor(1.),_epsilon_std_0.05}.pt"
model_config = load_model_params("train/unitary_single_qubit_gate/model_params.json")


pipeline = Pipeline(
    UniversalQOCTransformer(**model_config),
    weight_path = weight_path,
    device="cpu"
)


train_set_name = [
    "X_gate",
    "X(pi-2)_gate",
    "Y_gate",
    "Z(pi-4)_gate",
    "H_gate",
]

train_set = torch.stack(
    [
        torch.tensor([1, 0, 0, np.pi]),
        torch.tensor([1, 0, 0, np.pi/2]),
        torch.tensor([0, 1, 0, np.pi]),
        torch.tensor([0, 0, 1, np.pi/4]),
        torch.tensor([1, 0, 1, np.pi])
    ]
)

pulses = pipeline(train_set)

print(pulses.shape)

phase_control_only = True
PSI_INIT = torch.tensor([1.0, 0.0], dtype=torch.complex64)
model_name = "UniversalQOCTransformer"
save_dir = "figures/universal_gates/"
os.makedirs(save_dir, exist_ok=True)
y_labels = [r"Phase (units of $\pi$)"]


for target_name, rotation_vector, pulse in zip(train_set_name, train_set, pulses):
    print(f"Figures for {target_name}")

    U_target = rotation_to_unitary(rotation_vector.numpy())
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
    # deltas = [-2, -1, -0.5, 0, 0.5, 1, 2]
    # epsilons = [0]
    M = 11
    errors = get_ore_ple_error_distribution(batch_size=M)
    deltas, epsilons = errors[0], errors[1]
    # # uniform dist
    deltas = [-1 + 0.2 * i for i in range(M)]

    bloch_list, pulse_info_list, fidelity_list = [], [], []

    # target state
    target_psi = U_target @ PSI_INIT

    # for eps in epsilons:
    #     for delt in deltas:
    for eps, delt in zip(epsilons, deltas):
        # simulate
        psi = PSI_INIT
        bv, pi = [], []
        # tau = 0
        for p in df.itertuples():
            g = generate_unitary
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