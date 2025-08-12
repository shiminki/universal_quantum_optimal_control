import gradio as gr
import torch
import sys, os, glob, math, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")

# Add parent directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# Local imports
from train.unitary_single_qubit_gate.unitary_single_qubit_gate import (
    load_model_params
)
from model.universal_model import UniversalQOCTransformer, Pipeline
from visualize.util import (
    fidelity_contour_plot,
    plot_pulse_param,
    plot_fidelity_by_std,
    get_ore_ple_error_distribution,
    animate_multi_error_bloch
)

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


# Core compute: returns pulse tensor and target unitary
def compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw):
    # Model params
    if model_option == "100 length":
        name = "transformer_len100"
        path = "demo_universal/weight/length_100.pt"
        params = load_model_params("demo_universal/params/length_100.json")
    else:
        name = "transformer_len400"
        path = "demo_universal/weight/length_400.pt"
        params = load_model_params("demo_universal/params/length_400.json")
    axis = np.array([x_, y_, z_]); axis = axis / np.linalg.norm(axis)
    n_x, n_y, n_z = axis; theta = math.pi * theta_raw
    H = 0.5 * theta * (n_x*_SIGMA_X_CPU + n_y*_SIGMA_Y_CPU + n_z*_SIGMA_Z_CPU)
    U_target = torch.matrix_exp(-1j * H)

    pipeline = Pipeline(
        UniversalQOCTransformer(**params),
        weight_path=path,
        device="cpu"
    )

    pulse = pipeline(torch.tensor([n_x, n_y, n_z, theta], dtype=torch.float32).unsqueeze(0)).squeeze(0)
    return pulse, U_target

# 1. Compute pulse and CSV
def run_params(model_option, x_, y_, z_, theta_raw):
    pulse, _ = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)
    df = pd.DataFrame(pulse.numpy(), columns=["phi", "tau"])
    outdir = os.path.join("demo_outputs","params")
    os.makedirs(outdir,exist_ok=True)
    path = os.path.join(outdir,"pulse_params.csv"); df.to_csv(path,index=False)
    return df, path

# 2. Contour plot
def run_contour(model_option, x_, y_, z_, theta_raw):
    pulse, U = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)
    outdir = os.path.join("demo_outputs","contour")
    os.makedirs(outdir,exist_ok=True)
    fidelity_contour_plot("tgt",U,pulse,model_option,outdir,phase_only=True)
    imgs = sorted(glob.glob(os.path.join(outdir,"*.png")))
    return imgs

# 3. Pulse param plot
def run_paramplot(model_option, x_, y_, z_, theta_raw,):
    target_name = f"axis=({x_:.3f}, {y_:.3f}, {z_:.3f}), theta={theta_raw:.3f} pi"

    pulse, U = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)
    df = pd.DataFrame(pulse.numpy())
    outdir = os.path.join("demo_outputs","paramplot")
    os.makedirs(outdir,exist_ok=True)
    plot_pulse_param(
        outdir, target_name, ["Phase (units of pi)"], df
    )

    imgs = sorted(glob.glob(os.path.join(outdir,"*.png")))
    return imgs

# 4. Fidelity vs std
def run_fidelity(model_option, x_, y_, z_, theta_raw):
    pulse, U = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)
    outdir = os.path.join("demo_outputs","fidelity_std")
    os.makedirs(outdir,exist_ok=True)
    plot_fidelity_by_std("tgt",U,pulse,model_option,outdir,phase_only=True)
    imgs = sorted(glob.glob(os.path.join(outdir,"*.png")))
    return imgs

# 5. Evolution video
def run_evolution(model_option, x_, y_, z_, theta_raw):
    pulse, U_target = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)

    df = pd.DataFrame(pulse.numpy(), columns=["phi", "tau"])

    target_name = f"axis=({x_:.3f}, {y_:.3f}, {z_:.3f}), theta={theta_raw:.3f} pi"
    
    M = 11
    errors = get_ore_ple_error_distribution(batch_size=M)
    deltas, epsilons = errors[0], errors[1]
    # # uniform dist
    # deltas = np.random.random(M) * 2 - 1
    deltas = [-1 + 0.2 * i for i in range(M)]

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
    

    outdir=os.path.join("demo_outputs","evolution"); os.makedirs(outdir,exist_ok=True)
    vid=os.path.join(outdir,"evolution.mp4")

    os.makedirs(outdir, exist_ok=True)
    animate_multi_error_bloch(
        bloch_list, pulse_info_list, fidelity_list,
        deltas, epsilons,
        name=f"Ensemble Evolution of {target_name}",
        save_path=vid,
        phase_only=True
    )

    return vid, [vid]

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Composite Pulse for Universal Gates")
    with gr.Row():
        c1,c2 = gr.Column(scale=1), gr.Column(scale=2)
    with c1:
        model = gr.Dropdown(["100 length","400 length"],value="100 length",label="Model")
        x_in=gr.Number(1.0,label="x-component"); y_in=gr.Number(0.0,label="y-component"); z_in=gr.Number(0.0,label="z-component")
        th=gr.Slider(0.0,2.0,value=0.5,step=0.01,label="Theta (π units)")
        btn1=gr.Button("Compute Parameters")
        btn2=gr.Button("Fidelity vs Std"); btn3=gr.Button("Evolution Video")
    with c2:
        df_out=gr.Dataframe(label="Pulse Params CSV"); csv_dl=gr.File(label="Download CSV")
        cont_gallery=gr.Gallery(label="Contour"); param_gallery=gr.Gallery(label="Param Plot")
        fid_gallery=gr.Gallery(label="Fidelity vs Std"); vid_out=gr.Video(label="Evolution"); vid_dl=gr.File(label="Download Video")
    btn1.click(run_params,[model,x_in,y_in,z_in,th],[df_out,csv_dl])
    btn1.click(run_contour,[model,x_in,y_in,z_in,th],[cont_gallery])
    btn1.click(run_paramplot,[model,x_in,y_in,z_in,th],[param_gallery])
    btn2.click(run_fidelity,[model,x_in,y_in,z_in,th],[fid_gallery])
    btn3.click(run_evolution,[model,x_in,y_in,z_in,th],[vid_out,vid_dl])
if __name__ == "__main__":
    demo.launch(share=True)
