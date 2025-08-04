import gradio as gr
import torch
import sys, os, glob, math, numpy as np, pandas as pd

# Add parent directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# Local imports
from train.unitary_single_qubit_gate.unitary_single_qubit_gate import (
    load_model_params,
    CompositePulseTransformerEncoder,
    get_score_emb_unitary
)
from visualize.util import (
    fidelity_contour_plot,
    plot_pulse_param,
    plot_fidelity_by_std,
    get_ore_ple_error_distribution,
    animate_multi_error_bloch
)
from demo_util import (
    generate_unitary,
    spinor_to_bloch,
    euler_yxy_from_axis_angle,
    Ry, Rx,
    _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU
)

# Core compute: returns pulse tensor and target unitary
def compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw):
    # Model params
    if model_option == "100 length":
        name = "composite_pulse_len100"
        path = "demo/weight/length_100.pt"
        params = load_model_params("demo/params/length_100.json")
    else:
        name = "composite_pulse_len400"
        path = "demo/weight/length_400.pt"
        params = load_model_params("demo/params/length_400.json")
    axis = np.array([x_, y_, z_]); axis = axis / np.linalg.norm(axis)
    n_x, n_y, n_z = axis; theta = math.pi * theta_raw
    H = 0.5 * theta * (n_x*_SIGMA_X_CPU + n_y*_SIGMA_Y_CPU + n_z*_SIGMA_Z_CPU)
    U_target = torch.matrix_exp(-1j * H)

    def get_p(angle, phase):
        m = CompositePulseTransformerEncoder(**params)
        m.load_state_dict(torch.load(path)); m.eval()
        base, _ = get_score_emb_unitary(0, angle)
        p = m(base.unsqueeze(0)).squeeze(0).detach(); p[:,0] += phase
        return p

    # Decomposition
    if n_z != 0:
        a,b,g = euler_yxy_from_axis_angle(n_x,n_y,n_z,theta); pulses=[]
        if g!=0: pulses.append(get_p(abs(g), math.pi/2 if g>0 else 3*math.pi/2))
        if b!=0: pulses.append(get_p(b,0.0))
        if a!=0: pulses.append(get_p(abs(a), math.pi/2 if a>0 else 3*math.pi/2))
        pulse = torch.cat(pulses,dim=0)
    else:
        phi = math.atan2(n_y,n_x)
        if theta>=math.pi: theta=2*math.pi-theta; phi+=math.pi
        pulse = get_p(theta,phi)
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
