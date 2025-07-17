import streamlit as st
import torch

import sys
import os
import glob


# Add parent directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))


from train.unitary_single_qubit_gate.unitary_single_qubit_gate import *
from visualize.util import *

# Configure the Streamlit app
st.set_page_config(page_title="Composite Pulse for Universal Gates", layout="centered")

# Title of the app
st.title("Composite Pulse for Universal Gates")

# User inputs for the two parameters
phi_raw = st.number_input("Phi (units of pi from -1 to 1)", value=0.0, format="%.2f")
theta_raw = st.number_input("Theta (units of pi from 0 to 1)", value=0.0, format="%.2f")

st.write("---")
st.write("### Inputs")
st.write(f"**Phi (units of pi):** {phi_raw}")
st.write(f"**Theta (units of pi):** {theta_raw}")

#######################################################
# Qubit Ensemble Evolution Video Helper Functions #####
#######################################################


_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]


def generate_unitary(pulse, delta, epsilon):
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

#######################################


# Button to run inference
if st.button("Run Inference"):
    phi = math.pi * phi_raw
    theta = math.pi * theta_raw
    # TODO: Replace the placeholder with your actual inference code
   
    # Load model parameters from external JSON
    model_params = load_model_params("train/unitary_single_qubit_gate/model_params.json")
    model = CompositePulseTransformerEncoder(**model_params)

    # load pretrained module

    model_path = "weights/universal_gate_length_100_finetune/err_{_delta_std_tensor(1.),_epsilon_std_0.05}.pt"
    model_name = "100 Pulse Model"
    model.load_state_dict(torch.load(model_path))
    model.eval()


    SCORE_tensor, U_target = get_score_emb_unitary(phi, theta)

    pulse = model(SCORE_tensor.unsqueeze(0)).squeeze(0).detach()

    pulse[:, 0] += phi

    target_name = f"phi={phi_raw:.2f}pi, theta={theta_raw:.2f}pi"

    save_dir = f"demo/dump/100_length_finetuned{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load and save plot param csv
    df = pd.DataFrame(pulse)
    csv_dir = os.path.join(save_dir, "pulse_param_csv")
    os.makedirs(csv_dir, exist_ok=True)
    pulse_csv_dir = os.path.join(csv_dir, f"{target_name}_pulse.csv")
    df.to_csv(pulse_csv_dir, index=False)

    # Display CSV and provide download
    st.write("### Pulse Parameters CSV")
    st.dataframe(df)
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name=f"{target_name}_pulse.csv",
        mime="text/csv"
    )

    # 2. Fidelity contour plot
    contour_dir = os.path.join(save_dir, "fidelity_contour_plot")
    os.makedirs(contour_dir, exist_ok=True)
    fidelity_contour_plot(
        target_name, U_target, pulse, model_name,
        contour_dir, phase_only=True
    )
    # Display contour images
    for img_path in glob.glob(os.path.join(contour_dir, "*.png")):
        st.write("### Fidelity Contour")
        st.image(img_path)

    # 3. Pulse parameter plot
    param_dir = os.path.join(save_dir, "pulse_param")
    os.makedirs(param_dir, exist_ok=True)
    plot_pulse_param(
        param_dir, target_name, ["Phase (units of pi)"], df
    )
    for img_path in glob.glob(os.path.join(param_dir, "*.png")):
        st.write("### Pulse Parameter Plot")
        st.image(img_path)

    # 4. Fidelity vs std(delta)
    std_dir = os.path.join(save_dir, "fidelity_vs_delta_std")
    os.makedirs(std_dir, exist_ok=True)
    plot_fidelity_by_std(
        target_name, U_target, pulse, model_name,
        std_dir, phase_only=True
    )
    for img_path in glob.glob(os.path.join(std_dir, "*.png")):
        st.write("### Fidelity vs Delta Std")
        st.image(img_path)




    # 5. Qubit evolution video
    M = 30
    errors = get_ore_ple_error_distribution(batch_size=M)
    deltas, epsilons = errors[0], errors[1]
    # # uniform dist
    deltas = np.random.random(M) * 2 - 1

    bloch_list, pulse_info_list, fidelity_list = [], [], []

    # target state
    PSI_INIT = torch.tensor([1, 0], dtype=torch.cfloat)
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
            
            # tau += p[2]
            pi.append((0, p[1], p[2]))
            

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
        phase_only=True
    )
    # Display video
    st.write("### Qubit Evolution Video")
    for vid_path in glob.glob(os.path.join(video_dir, "*.mp4")):
        st.video(vid_path)

    st.success("Inference complete and results displayed below.")

