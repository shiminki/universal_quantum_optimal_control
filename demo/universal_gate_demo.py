import streamlit as st
import torch

import sys
import os
import glob
from scipy.linalg import expm
from scipy.optimize import minimize


# Add parent directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))


from train.unitary_single_qubit_gate.unitary_single_qubit_gate import *
from visualize.util import *


#######################################################
#  Helper Functions #####
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


def Rx(theta):
    return expm(-1j * _SIGMA_X_CPU * theta / 2)

def Ry(theta):
    return expm(-1j * _SIGMA_Y_CPU * theta / 2)

def decompose_SU2(U_target):
    def loss(params):
        alpha, beta, gamma = params
        U = Rx(alpha) @ Ry(beta) @ Rx(gamma)
        return np.linalg.norm(U - U_target)

    res = minimize(loss, x0=[0.0, 0.0, 0.0])
    return res.x  # returns alpha, beta, gamma


def euler_yxy_from_axis_angle(nx, ny, nz, theta, *, eps=1e-12):
    """
    Return Euler angles (alpha, beta, gamma) for the y‑x‑y sequence so that
        R_y(alpha) R_x(beta) R_y(gamma)
    equals a rotation of angle `theta` about the axis (nx, ny, nz).

    Handles every special case: θ≈0, θ≈π, axis along ±y, axis in x‑z plane.
    """
    # ---- normalise the axis ------------------------------------------------
    n = np.array([nx, ny, nz], dtype=float)
    n /= np.linalg.norm(n)
    nx, ny, nz = n

    # ---- useful trig shorthands -------------------------------------------
    c = np.cos(theta)          # cos θ
    s = np.sin(theta)          # sin θ
    k = 1.0 - c                # = 2 sin²(θ/2)

    # ---- middle angle β ----------------------------------------------------
    cos_beta = np.clip(c + k * ny * ny, -1.0, 1.0)
    beta     = np.arccos(cos_beta)
    sin_beta = np.sin(beta)

    # -----------------------------------------------------------------------
    # 1) Generic case  (sin β not tiny)  →  use the closed‑form directly
    # -----------------------------------------------------------------------
    if abs(sin_beta) > eps:
        alpha = np.arctan2(k * nx * ny - s * nz,
                           k * ny * nz + s * nx)
        gamma = np.arctan2(k * nx * ny + s * nz,
                           s * nx       - k * ny * nz)
        return alpha, beta, gamma

    # -----------------------------------------------------------------------
    # 2) Singular cases  (sin β ≈ 0)  →  β is 0 or π
    # -----------------------------------------------------------------------
    if abs(ny) > 1 - eps:          # axis almost parallel to ±y
        # Rotation sits entirely on the y‑axis
        alpha = 0.0
        beta  = 0.0
        gamma = +theta if ny > 0 else -theta
        return alpha, beta, gamma

    # Remaining possibility:  ny ≈ 0  and  β ≈ π  (axis in x‑z plane,
    # half‑turn).  Keep α = 0 and solve γ from the x–z block.
    alpha = 0.0
    beta  = np.pi
    gamma = np.arctan2(k * nx * nz,          #  2 nx nz   when θ = π
                       c + k * nx * nx)      # -1+2 nx²   when θ = π
    return alpha, beta, gamma

#######################################




# Configure the Streamlit app
st.set_page_config(page_title="Composite Pulse for Universal Gates", layout="centered")

# Title of the app
st.title("Composite Pulse for Universal Gates")

# Model selection dropdown
model_option = st.selectbox(
    "Select model length:",
    options=["100 length", "400 length"]
)
# Map selection to model_name and model_path
if model_option == "100 length":
    model_name = "composite_pulse_len100"
    model_path = "demo/weight/length_100.pt"
    model_params = load_model_params("demo/params/length_100.json")
else:
    model_name = "composite_pulse_len400"
    model_path = "demo/weight/length_400.pt"
    model_params = load_model_params("demo/params/length_400.json")

# User inputs for the two parameters


# Description with better formatting
u_target_text = r"""
### Specify the target unitary

Enter the rotation axis $\mathbf{n} = (x, y, z)$ and the rotation angle $\theta$.  
The target unitary is:  

$\displaystyle U_{\text{target}} = e^{-i \, \mathbf{n} \cdot \boldsymbol{\sigma} \, \theta /2}$
"""

st.markdown(u_target_text)

# Axis inputs in a horizontal layout
col1, col2, col3 = st.columns(3)
with col1:
    x_ = st.number_input("x-component", value=1.0, format="%.3f")
with col2:
    y_ = st.number_input("y-component", value=0.0, format="%.3f")
with col3:
    z_ = st.number_input("z-component", value=0.0, format="%.3f")

theta_raw = st.slider(
    "Theta (in units of π)", min_value=0.0, max_value=2.0, value=0.5, step=0.01
)

assert 0 <= theta_raw <= 2, "theta out of range"

norm = np.sqrt(x_**2 + y_**2 + z_**2)
axis = np.array([x_, y_, z_]) / norm

n_x, n_y, n_z = axis
theta = math.pi * theta_raw

H = 0.5 * theta * (n_x * _SIGMA_X_CPU + n_y * _SIGMA_Y_CPU + n_z * _SIGMA_Z_CPU)
U_target = torch.matrix_exp(-1j * H)


# --- Outputs ---
st.write("---")
st.subheader("Inputs")
st.write(f"User-input axis: **{axis}**, θ = **{theta_raw:.3f} π**")

# Pretty matrix output using LaTeX
U_latex = r"\begin{bmatrix}" + \
          f"{U_target[0,0]:.3f} & {U_target[0,1]:.3f} \\\\ " + \
          f"{U_target[1,0]:.3f} & {U_target[1,1]:.3f}" + \
          r"\end{bmatrix}"
st.latex(U_latex)



# Button to run inference
if st.button("Run Inference"):
    
    def get_pulse(theta, phi):
        theta = math.pi * np.round(theta/math.pi, 3)

        model = CompositePulseTransformerEncoder(**model_params)

        # load pretrained module

        model.load_state_dict(torch.load(model_path))
        model.eval()

        _, U_target = get_score_emb_unitary(phi, theta)
        SCORE_tensor_base, _ = get_score_emb_unitary(0, theta)

        pulse = model(SCORE_tensor_base.unsqueeze(0)).squeeze(0).detach()

        pulse[:, 0] += phi
        return pulse
    

    decompose_msg = (
        r"Decomposing U_target into $U = R_y(\alpha) R_x(\beta) R_y(\gamma)$"
    )

    st.write(decompose_msg)

    alpha, beta, gamma = euler_yxy_from_axis_angle(n_x, n_y, n_z, theta)

    U_out = Ry(alpha) @ Rx(beta) @ Ry(gamma)

    result_msg = (
        fr"$U = R_y({alpha:.3f}) R_x({beta:.3f}) R_y({gamma:.3f})$"
    )

    st.write(result_msg)

    # Pretty matrix output using LaTeX
    U_out_latex = r"\begin{bmatrix}" + \
            f"{U_out[0,0]:.3f} & {U_out[0,1]:.3f} \\\\ " + \
            f"{U_out[1,0]:.3f} & {U_out[1,1]:.3f}" + \
            r"\end{bmatrix}"
    st.latex(U_out_latex)


    target_name = f"axis=({n_x:.3f}, {n_y:.3f}, {n_z:.3f}), theta={theta_raw:.3f} pi"

    save_dir = f"demo/dump/{model_option}_finetuned{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    pulses = []

    if gamma != 0:
        pulses.append(get_pulse(gamma, np.pi/2))

    if beta != 0:
        pulses.append(get_pulse(beta, 0))

    if alpha != 0:
        pulses.append(get_pulse(alpha, np.pi/2))

    pulse = torch.cat(pulses, dim=0)


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

