import gradio as gr
import torch
import sys, os, glob, math, numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
matplotlib.use("Agg")


# Local imports
from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import (
    load_model_params
)
from model.universal_model import UniversalQOCTransformer, Pipeline
from model.GRAPE_model import GRAPE
from smoothing.smooth_pulse import smooth_pulse
from visualize.util import (
    fidelity_contour_plot,
    plot_pulse_param,
    plot_fidelity_by_std,
    get_ore_ple_error_distribution,
    animate_multi_error_bloch
)
from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import (
    batched_unitary_generator,
    fidelity as compute_fidelity,
)

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
pauli = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]


from huggingface_hub import hf_hub_download  # pip install huggingface_hub

WEIGHTS_REPO = os.getenv("WEIGHTS_REPO", "shiminki/universal_qoc_weights")

MODEL_SELECTION = {
    "100 length": {
        "name": "transformer_len100",
        "filename": "length_100.pt",
        "weight": "demo_universal/weight/length_100.pt",
        "params": "demo_universal/params/length_100.json",
        "type": "transformer",
    },
    "400 length": {
        "name": "transformer_len400",
        "filename": "length_400.pt",
        "weight": "demo_universal/weight/length_400.pt",
        "params": "demo_universal/params/length_400.json",
        "type": "transformer",
    },
    "GRAPE": {
        "name": "GRAPE",
        "filename": "grape.pt",
        "weight": "demo_universal/weight/grape.pt",
        "params": "demo_universal/params/grape.json",
        "type": "grape",
    },
}

def _resolve_weight_path(model_option, revision=None):
    meta = MODEL_SELECTION[model_option]
    return hf_hub_download(
        repo_id=WEIGHTS_REPO,
        filename=meta["filename"],
        revision=revision,               # Optional: pin a specific commit/tag for reproducibility
    )



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

name = "initial"

# Core compute: returns pulse tensor and target unitary
def compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw):
    meta = MODEL_SELECTION[model_option]
    name = meta["name"]
    params = load_model_params(meta["params"])
    # path = _resolve_weight_path(model_option)  # <- from HF model repo
    # path = meta["weight"]  # <- local path for testing

    path = _resolve_weight_path(model_option) if args.hf_weight else meta["weight"]
    
    axis = np.array([x_, y_, z_]); axis = axis / np.linalg.norm(axis)
    n_x, n_y, n_z = axis; theta = math.pi * theta_raw
    H = 0.5 * theta * (n_x*_SIGMA_X_CPU + n_y*_SIGMA_Y_CPU + n_z*_SIGMA_Z_CPU)
    U_target = torch.matrix_exp(-1j * H)


    if meta["type"] == "transformer":
        pipeline = Pipeline(
            UniversalQOCTransformer(**params),
            weight_path=path,
        )
        pulse = pipeline(torch.tensor([n_x, n_y, n_z, theta], dtype=torch.float32).unsqueeze(0)).squeeze(0)
    else:
        grape = GRAPE(**params)
        # grape = GRAPE(**params)
        grape.load_state_dict(torch.load(path, map_location="cpu"))

        U = torch.matrix_exp(
            -1j * (n_x * _SIGMA_X_CPU + n_y * _SIGMA_Y_CPU + n_z * _SIGMA_Z_CPU) * theta / 2
        )

        # pulse = grape(U.unsqueeze(0)).squeeze(0).detach()
        pulse = grape(torch.tensor([n_x, n_y, n_z, theta], dtype=torch.float32).unsqueeze(0)).squeeze(0).detach()

        print("pulse.shape", pulse.shape)

    return pulse, U_target

# Helper function that returns azimuthal and polar angles from an unit vector

def get_angles_from_axis(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0.0  # polar angle
    phi = np.arctan2(y, x)                      # azimuthal angle
    return phi, theta


def csv_to_internal_pulse(df, Omega):
    """Convert CSV-format pulse (t_us, Omega_x, Omega_y) back to internal (phi, tau) format."""
    Omega_x = df["Omega_x (2pi MHz)"].values
    Omega_y = df["Omega_y (2pi MHz)"].values
    t_us = df["t (us)"].values
    phi = np.arctan2(Omega_y, Omega_x)
    Omega_angular = 2 * math.pi * Omega
    tau = t_us * Omega_angular
    return torch.tensor(np.stack([phi, tau], axis=1), dtype=torch.float32)



# 1. Compute pulse and CSV
def run_params(model_option, x_, y_, z_, theta_raw, Omega, Omega_cutoff):
    pulse, _ = compute_pulse_and_unitary(model_option, x_, y_, z_, theta_raw)
    Omega_angular = 2 * math.pi * Omega  # Convert to angular frequency in MHz
    phi, tau = pulse[:,0].numpy(), pulse[:,1].numpy() / Omega_angular  # Convert tau to microseconds using Omega
    H_x = np.cos(phi) * Omega  # H_x in units of 2pi MHz
    H_y = np.sin(phi) * Omega  # H_y in units of 2pi MHz
    H_z = np.zeros_like(H_x)    # H_z is zero for this control scheme
    pulse_np = np.stack([tau, H_x, H_y, H_z], axis=1)  # tau in microseconds, H in units of 2pi MHz
    df = pd.DataFrame(pulse_np, columns=["t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"])
    outdir = os.path.join("demo_outputs","params")
    os.makedirs(outdir,exist_ok=True)
    path = os.path.join(outdir,"pulse_params.csv"); df.to_csv(path,index=False)

    # Smooth the pulse
    smooth_path = os.path.join(outdir, "pulse_params_smoothed.csv")
    smooth_pulse(path, smooth_path, cutoff_MHz=Omega_cutoff, omega_max=Omega)
    df_smooth = pd.read_csv(smooth_path)

    return df, path, df_smooth, smooth_path

# 2. Contour plot
def run_contour(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff):
    pulse, U = compute_pulse_and_unitary(model_option, x_, y_, z_, lambda_raw)
    outdir = os.path.join("demo_outputs","contour")
    os.makedirs(outdir,exist_ok=True)
    phi, theta = get_angles_from_axis(x_, y_, z_)
    target_name = f"(axis=n(phi={phi:.3f}, theta={theta:.3f}), rotation={lambda_raw:.3f} pi)"

    # Original contour
    fidelity_contour_plot(target_name,U,pulse,model_option,outdir,phase_only=True, Omega=Omega)

    # Smoothed contour
    _, _, df_smooth, _ = run_params(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff)
    pulse_smooth = csv_to_internal_pulse(df_smooth, Omega)
    smooth_name = f"{target_name} [Smoothed {int(Omega_cutoff)} MHz]"
    fidelity_contour_plot(smooth_name, U, pulse_smooth, model_option, outdir, phase_only=True, Omega=Omega)

    imgs = sorted(glob.glob(os.path.join(outdir, f"{target_name}*")))
    return imgs

# 3. Pulse param plot
def run_paramplot(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff):
    phi, theta = get_angles_from_axis(x_, y_, z_)
    target_name = f"axis=n(phi={phi:.3f}, theta={theta:.3f}), lambda={lambda_raw:.3f} pi, {model_option}"

    df, _, df_smooth, _ = run_params(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff)
    outdir = os.path.join("demo_outputs","paramplot")
    os.makedirs(outdir,exist_ok=True)
    plot_pulse_param(
        outdir, target_name, df, Omega, df_smooth=df_smooth,
        phi=phi, theta=theta, lam=lambda_raw, model=model_option,
        cutoff=Omega_cutoff,
    )

    imgs = glob.glob(os.path.join(outdir, f"{target_name}.png"))
    return imgs

# 4. Fidelity vs std
def run_fidelity(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff):
    pulse, U = compute_pulse_and_unitary(model_option, x_, y_, z_, lambda_raw)
    outdir = os.path.join("demo_outputs","fidelity_std")
    os.makedirs(outdir,exist_ok=True)
    phi, theta = get_angles_from_axis(x_, y_, z_)
    target_name = f"axis=n(phi={phi:.3f}, theta={theta:.3f}), lambda={lambda_raw:.3f} pi"

    # Original fidelity
    if not os.path.exists(os.path.join(outdir, f"{target_name}_fidelity.png")):
        plot_fidelity_by_std(target_name,U,pulse,model_option,outdir,phase_only=True)

    # Smoothed fidelity
    _, _, df_smooth, _ = run_params(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff)
    pulse_smooth = csv_to_internal_pulse(df_smooth, Omega)
    smooth_name = f"{target_name} [Smoothed {int(Omega_cutoff)} MHz]"
    if not os.path.exists(os.path.join(outdir, f"{smooth_name}_fidelity.png")):
        plot_fidelity_by_std(smooth_name, U, pulse_smooth, model_option, outdir, phase_only=True)

    imgs = sorted(glob.glob(os.path.join(outdir, f"{target_name}*")))

    return imgs

# 5. Evolution video
def run_evolution(model_option, x_, y_, z_, lambda_raw, Omega):
    pulse, U_target = compute_pulse_and_unitary(model_option, x_, y_, z_, lambda_raw)

    df = pd.DataFrame(pulse.numpy(), columns=["phi", "tau"])

    target_name = f"axis=n(phi={get_angles_from_axis(x_, y_, z_)[0]:.3f}, theta={get_angles_from_axis(x_, y_, z_)[1]:.3f}), lambda={lambda_raw:.3f} pi, \nModel:{model_option}"
    outdir=os.path.join("demo_outputs","evolution"); os.makedirs(outdir,exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    vid=os.path.join(outdir, f"{target_name}_evolution.mp4")

    if not os.path.exists(vid):
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
    
        animate_multi_error_bloch(
            bloch_list, pulse_info_list, fidelity_list,
            deltas, epsilons,
            name=f"Ensemble Evolution of {target_name}",
            save_path=vid,
            phase_only=True,
            Omega=Omega
        )

    return vid, [vid]

# 6. Fidelity vs Cutoff
def run_fidelity_vs_cutoff(model_option, x_, y_, z_, lambda_raw, Omega, Omega_cutoff_val,
                           cutoff_min=80.0, cutoff_max=8000.0, cutoff_step=80.0):

    pulse, U_target = compute_pulse_and_unitary(model_option, x_, y_, z_, lambda_raw)

    # Cutoff values in (2pi) MHz
    cutoffs = np.arange(cutoff_min, cutoff_max + cutoff_step / 2, cutoff_step)

    M = 5000  # Monte Carlo samples
    # delta ~ Rabi * N(0,1), epsilon ~ N(0, 0.05)
    errors_mc = torch.stack([
        torch.randn(M) * 1.0,   # delta_std = 1 (in units of Omega)
        torch.randn(M) * 0.05,  # epsilon_std = 0.05
    ])

    U_target_batch = U_target.unsqueeze(0).expand(M, -1, -1)
    pulse_batch_orig = pulse.unsqueeze(0).expand(M, -1, -1)

    # Original (unfiltered) fidelity
    U_out_orig = batched_unitary_generator(pulse_batch_orig, errors_mc)
    F_orig = compute_fidelity(U_out_orig, U_target_batch, 1).mean().item()

    infidelities = []
    outdir = os.path.join("demo_outputs", "params")
    os.makedirs(outdir, exist_ok=True)

    # Prepare the original CSV once
    Omega_angular = 2 * math.pi * Omega
    phi_pulse, tau_pulse = pulse[:, 0].numpy(), pulse[:, 1].numpy() / Omega_angular
    H_x = np.cos(phi_pulse) * Omega
    H_y = np.sin(phi_pulse) * Omega
    H_z = np.zeros_like(H_x)
    pulse_np = np.stack([tau_pulse, H_x, H_y, H_z], axis=1)
    df_orig = pd.DataFrame(pulse_np, columns=["t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"])
    orig_csv = os.path.join(outdir, "pulse_params_cutoff_sweep.csv")
    df_orig.to_csv(orig_csv, index=False)

    for cutoff in cutoffs:
        smooth_csv = os.path.join(outdir, "pulse_smoothed_cutoff_sweep.csv")
        smooth_pulse(orig_csv, smooth_csv, cutoff_MHz=cutoff, omega_max=Omega)
        df_smooth = pd.read_csv(smooth_csv)
        pulse_smooth = csv_to_internal_pulse(df_smooth, Omega)

        pulse_batch = pulse_smooth.unsqueeze(0).expand(M, -1, -1)
        U_out = batched_unitary_generator(pulse_batch, errors_mc)
        F_mean = compute_fidelity(U_out, U_target_batch, 1).mean().item()
        infidelities.append(1.0 - F_mean)

    infidelities = np.array(infidelities)
    # Clamp tiny values for log-log
    infidelities = np.clip(infidelities, 1e-12, None)

    # Power-law fit: infidelity ~ cutoff^eta  =>  log(inf) = eta*log(cutoff) + C
    log_c = np.log(cutoffs)
    log_inf = np.log(infidelities)
    # Filter out any -inf or nan
    valid = np.isfinite(log_c) & np.isfinite(log_inf)
    if valid.sum() >= 2:
        coeffs = np.polyfit(log_c[valid], log_inf[valid], 1)
        eta = coeffs[0]
        C = coeffs[1]
        fit_inf = np.exp(eta * log_c + C)
    else:
        eta = float('nan')
        fit_inf = None

    # Plot
    phi, theta = get_angles_from_axis(x_, y_, z_)
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.loglog(cutoffs, infidelities, 'o-', color='C0', label='Infidelity')

    ax.plot(cutoffs, infidelities, 'o-', color='C0', label='Infidelity')

    ax.axhline(y=1-F_orig, color='r', linestyle='--', label="Unfiltered Infidelity")

    # if fit_inf is not None:
    #     ax.loglog(cutoffs, fit_inf, color='orange', linestyle='--',
    #               label=fr'Fit: $1-F \propto \omega_c^{{{eta:.2f}}}$')
    ax.set_xlabel(r"Cutoff frequency $(2\pi)$ MHz")
    ax.set_ylabel("Infidelity $(1 - F)$")
    ax.set_title(
        f"Infidelity vs Cutoff for "
        r"$U(\phi=$" + f"{phi:.3f}, " + r"$\theta=$" + f"{theta:.3f}, "
        + r"$\lambda=$" + f"{lambda_raw:.3f}" + r"$\pi)$"
        + f"\nmodel={model_option}, Rabi = $(2\\pi)$ {Omega} MHz"
        + fr", $\eta = {eta:.3f}$"
    )
    ax.semilogy()
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    plot_dir = os.path.join("demo_outputs", "fidelity_vs_cutoff")
    os.makedirs(plot_dir, exist_ok=True)
    target_name = f"cutoff_sweep_phi{phi:.3f}_theta{theta:.3f}_lam{lambda_raw:.3f}_{model_option}"
    out_path = os.path.join(plot_dir, f"{target_name}.png")
    plt.savefig(out_path)
    plt.close(fig)

    # --- Frequency spectrum: |F[Omega](f)| where Omega = Omega_x + i*Omega_y ---
    # Build piecewise-constant waveform on a fine uniform time grid
    t_segs = tau_pulse  # segment durations in microseconds
    T_total = t_segs.sum()  # total duration in us
    oversample = 8
    dt_u = t_segs.mean() / oversample  # uniform grid spacing in us
    N_grid = int(np.ceil(T_total / dt_u))
    t_edges = np.concatenate([[0.0], np.cumsum(t_segs)])

    # Fill uniform grid with piecewise-constant values
    Omega_x_uniform = np.zeros(N_grid)
    Omega_y_uniform = np.zeros(N_grid)
    for j in range(len(t_segs)):
        i0 = min(int(round(t_edges[j] / dt_u)), N_grid)
        i1 = min(int(round(t_edges[j + 1] / dt_u)), N_grid)
        if i1 > i0:
            Omega_x_uniform[i0:i1] = H_x[j]
            Omega_y_uniform[i0:i1] = H_y[j]

    # Complex control field and its FFT
    Omega_complex = Omega_x_uniform + 1j * Omega_y_uniform
    spectrum = np.fft.fft(Omega_complex)
    freqs_MHz = np.fft.fftfreq(N_grid, d=dt_u)  # frequencies in MHz (since dt_u is in us)
    amplitude_spectrum = np.abs(spectrum) / N_grid  # normalized amplitude

    # Plot spectrum
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(freqs_MHz, amplitude_spectrum, color='C0', linewidth=0.8)
    ax2.axvline(x=Omega_cutoff_val, color='r', linestyle='--', linewidth=1.5,
                label=f'Slider cutoff = $(2\\pi)$ {Omega_cutoff_val:.0f} MHz')
    ax2.set_xlabel(r"Frequency $(2\pi)$ MHz")
    ax2.set_ylabel(r"$|\mathcal{F}[\Omega](f)|$ $(2\pi$ MHz)")
    ax2.set_title(
        f"Frequency spectrum of pulse for "
        r"$U(\phi=$" + f"{phi:.3f}, " + r"$\theta=$" + f"{theta:.3f}, "
        + r"$\lambda=$" + f"{lambda_raw:.3f}" + r"$\pi)$"
        + f"\nmodel={model_option}, Rabi = $(2\\pi)$ {Omega} MHz, "
        + f"T = {T_total * 1000:.2f} ns"
    )
    ax2.set_xlim(0, 8000)
    ax2.semilogy()
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    spectrum_path = os.path.join(plot_dir, f"{target_name}_spectrum.png")
    plt.savefig(spectrum_path)
    plt.close(fig2)

    return [out_path, spectrum_path]


def _predownload_all(revision=None):
    for key in MODEL_SELECTION:
        _ = _resolve_weight_path(key, revision=revision)



# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Composite Pulse for Universal Gates")
    demo.load(lambda: _predownload_all(), outputs=[])
    
    with gr.Row():
        with gr.Column(scale=1) as c1:
            model = gr.Dropdown(["100 length", "400 length", "GRAPE"],value="400 length",label="Model")
            x_in=gr.Number(1.0,label="x-component"); y_in=gr.Number(0.0,label="y-component"); z_in=gr.Number(0.0,label="z-component")
            th=gr.Slider(0.0,2.0,value=0.5,step=0.01,label="Theta (π units)")
            Omega=gr.Slider(20, 100,value=80,step=1,label="Omega (Rabi freq in units of 2π MHz)")  # Optional: for future extension to variable Omega
            Omega_cutoff=gr.Slider(100, 10000, value=3000, step=100, label="Low-pass Filter Cutoff Frequency")
            
            btn1=gr.Button("Compute Parameters")
            btn2=gr.Button("Fidelity vs Std")
            btn3=gr.Button("Evolution Video")
            btn4=gr.Button("Fidelity vs Cutoff")

        with gr.Column(scale=2) as c2:
            df_out=gr.Dataframe(label="Original Pulse CSV"); csv_dl=gr.File(label="Download Original CSV")
            df_smooth_out=gr.Dataframe(label="Smoothed Pulse CSV"); csv_smooth_dl=gr.File(label="Download Smoothed CSV")
            cont_gallery=gr.Gallery(label="Contour (Original vs Smoothed)"); param_gallery=gr.Gallery(label="Param Plot (Original vs Smoothed)")
            fid_gallery=gr.Gallery(label="Fidelity vs Std"); vid_out=gr.Video(label="Evolution"); vid_dl=gr.File(label="Download Video")
            cutoff_gallery=gr.Gallery(label="Fidelity vs Cutoff")

        btn1.click(run_params,[model,x_in,y_in,z_in,th,Omega,Omega_cutoff],[df_out,csv_dl,df_smooth_out,csv_smooth_dl])
        btn1.click(run_contour,[model,x_in,y_in,z_in,th,Omega,Omega_cutoff],[cont_gallery])
        btn1.click(run_paramplot,[model,x_in,y_in,z_in,th,Omega,Omega_cutoff],[param_gallery])
        btn2.click(run_fidelity,[model,x_in,y_in,z_in,th,Omega,Omega_cutoff],[fid_gallery])
        btn3.click(run_evolution,[model,x_in,y_in,z_in,th,Omega],[vid_out,vid_dl])
        btn4.click(run_fidelity_vs_cutoff,[model,x_in,y_in,z_in,th,Omega,Omega_cutoff],[cutoff_gallery])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_weight", type=int, default=1, help="Loads huggingface weights")
    args = parser.parse_args()
    demo.launch(share=True)
