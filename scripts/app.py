"""Gradio demo for inspecting pretrained controllers on user-specified targets.

The UI mirrors the original app.py but delegates physics/models entirely to
`uqoc`. Weights come from HuggingFace by default (`--hf-weights 0` to use local).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch

from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running without `pip install -e .` (e.g. Google Colab)
_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
for _p in (_src, _root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uqoc.config import load_config
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import rotation_vector_to_unitary
from visualize import (
    animate_multi_error_bloch,
    fidelity_contour_plot,
    plot_fidelity_by_std,
    plot_pulse_param,
    spinor_to_bloch,
)


WEIGHTS_REPO = os.getenv("WEIGHTS_REPO", "shiminki/universal_qoc_weights")

MODELS = {
    "35 length":  {"config": "configs/transformer_len35.yaml",
                   "local": "weights/len35.pt",
                   "hf_name": "len35.pt",
                   "display": "transformer_len35"},
    "100 length": {"config": "demo_universal/config/length_100.yaml",
                   "local": "demo_universal/weight/length_100.pt",
                   "hf_name": "length_100.pt",
                   "display": "transformer_len100"},
    "400 length": {"config": "demo_universal/config/length_400.yaml",
                   "local": "demo_universal/weight/length_400.pt",
                   "hf_name": "length_400.pt",
                   "display": "transformer_len400"},
    "GRAPE":     {"config": "demo_universal/config/deep_nn.yaml",
                  "local": "demo_universal/weight/grape.pt",
                  "hf_name": "grape.pt",
                  "display": "deep_nn"},
}


def _weight_path(key: str, use_hf: bool) -> str:
    meta = MODELS[key]
    if use_hf:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=WEIGHTS_REPO, filename=meta["hf_name"])
    return meta["local"]


def _target_unitary(x: float, y: float, z: float, theta_pi: float) -> torch.Tensor:
    axis = np.array([x, y, z], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = math.pi * theta_pi
    rot_vec = torch.tensor([axis[0], axis[1], axis[2], theta], dtype=torch.float32)
    return rotation_vector_to_unitary(rot_vec)


def _pulse_and_unitary(key: str, x: float, y: float, z: float, theta_pi: float,
                       use_hf: bool) -> tuple[torch.Tensor, torch.Tensor]:
    meta = MODELS[key]
    cfg = load_config(meta["config"])
    model = build_model(cfg.model)
    pipe = Pipeline(model, weight_path=_weight_path(key, use_hf), device="cpu")

    axis = np.array([x, y, z], dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Existing transformer weights trained on theta < pi; nudge away from singularity
    theta = 0.99 * math.pi if abs(theta_pi - 1.0) < 1e-9 else math.pi * theta_pi
    rot_vec = torch.tensor([axis[0], axis[1], axis[2], theta], dtype=torch.float32).unsqueeze(0)
    pulse = pipe(rot_vec).squeeze(0)
    U_target = _target_unitary(x, y, z, theta_pi)
    return pulse, U_target


def _axis_angles(x: float, y: float, z: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    return math.atan2(y, x), math.acos(z / r) if r != 0 else 0.0


def _pulse_to_df(pulse: torch.Tensor, Omega: float) -> pd.DataFrame:
    Omega_ang = 2 * math.pi * Omega
    phi = pulse[:, 0].numpy()
    tau_us = pulse[:, 1].numpy() / Omega_ang
    H_x = np.cos(phi) * Omega
    H_y = np.sin(phi) * Omega
    H_z = np.zeros_like(H_x)
    return pd.DataFrame(
        np.stack([tau_us, H_x, H_y, H_z], axis=1),
        columns=["t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"],
    )


def _csv_to_internal(df: pd.DataFrame, Omega: float) -> torch.Tensor:
    phi = np.arctan2(df["Omega_y (2pi MHz)"].values, df["Omega_x (2pi MHz)"].values)
    tau = df["t (us)"].values * (2 * math.pi * Omega)
    return torch.tensor(np.stack([phi, tau], axis=1), dtype=torch.float32)


# ---------------------------------------------------------------- Gradio cbs
def _cb_params(key, x, y, z, th, Omega, cutoff):
    pulse, _ = _pulse_and_unitary(key, x, y, z, th, args.hf_weights)
    df = _pulse_to_df(pulse, Omega)
    os.makedirs("demo_outputs/params", exist_ok=True)
    orig = "demo_outputs/params/pulse_params.csv"
    df.to_csv(orig, index=False)
    from smoothing.smooth_pulse import smooth_pulse
    smooth = "demo_outputs/params/pulse_params_smoothed.csv"
    smooth_pulse(orig, smooth, cutoff_MHz=cutoff, omega_max=Omega)
    df_s = pd.read_csv(smooth)
    return df, orig, df_s, smooth


def _cb_contour(key, x, y, z, th, Omega, cutoff):
    pulse, U = _pulse_and_unitary(key, x, y, z, th, args.hf_weights)
    outdir = "demo_outputs/contour"
    phi, theta = _axis_angles(x, y, z)
    name = f"(axis=n(phi={phi:.2f}, theta={theta:.2f}), rotation={th:.2f} pi)"
    fidelity_contour_plot(name, U, pulse, key, outdir, phase_only=True, Omega=Omega)
    return sorted(glob.glob(os.path.join(outdir, f"{name}*")))


def _cb_paramplot(key, x, y, z, th, Omega, cutoff):
    phi, theta = _axis_angles(x, y, z)
    df, _, df_s, _ = _cb_params(key, x, y, z, th, Omega, cutoff)
    outdir = "demo_outputs/paramplot"
    title = f"axis=n(phi={phi:.2f}, theta={theta:.2f}), alpha={th:.2f} pi, {key}"
    plot_pulse_param(outdir, title, df, Omega, df_smooth=df_s,
                     phi=phi, theta=theta, lam=th, model=key, cutoff=cutoff)
    return glob.glob(os.path.join(outdir, f"{title}.png"))


def _cb_fidelity(key, x, y, z, th, Omega, cutoff):
    pulse, U = _pulse_and_unitary(key, x, y, z, th, args.hf_weights)
    outdir = "demo_outputs/fidelity_std"
    phi, theta = _axis_angles(x, y, z)
    name = f"axis=n(phi={phi:.2f}, theta={theta:.2f}), alpha={th:.2f} pi"
    plot_fidelity_by_std(name, U, pulse, key, outdir)
    return sorted(glob.glob(os.path.join(outdir, f"{name}*")))


def _cb_evolution(key, x, y, z, th, Omega):
    from uqoc.errors import ERROR_SAMPLERS
    pulse, U = _pulse_and_unitary(key, x, y, z, th, args.hf_weights)
    df = _pulse_to_df(pulse, Omega)
    phi, theta = _axis_angles(x, y, z)
    outdir = "demo_outputs/evolution"
    os.makedirs(outdir, exist_ok=True)
    target_name = f"axis=n(phi={phi:.2f}, theta={theta:.2f}), alpha={th:.2f} pi, Model:{key}"
    vid = os.path.join(outdir, f"{target_name}_evolution.mp4")

    M = 11
    deltas = [-1 + 0.2 * i for i in range(M)]
    epsilons = [0.0] * M
    PSI_INIT = torch.tensor([1.0, 0.0], dtype=torch.cfloat)
    target_psi = U @ PSI_INIT

    bloch_list, pulse_info_list, fidelity_list = [], [], []
    for eps, delt in zip(epsilons, deltas):
        psi = PSI_INIT
        bv, pi = [], []
        for _, tau_k, H_x_k, H_y_k, _ in df.itertuples():
            phi_k = math.atan2(H_y_k, H_x_k)
            tau_rad = tau_k * 2 * math.pi * Omega
            err = torch.tensor([[delt], [eps]], dtype=torch.float32)
            single = pulse[[0], :].clone()
            single[0, 0], single[0, 1] = phi_k, tau_rad
            step_U = batched_unitary_generator(single.unsqueeze(1), err).squeeze(0)
            psi = step_U @ psi
            bv.append(spinor_to_bloch(psi))
            pi.append((0, float(phi_k), float(tau_rad)))
        bloch_list.append(np.vstack(([spinor_to_bloch(PSI_INIT)], bv)))
        pulse_info_list.append(pi)
        fidelity_list.append(float(abs(torch.vdot(target_psi, psi)) ** 2))

    # animate_multi_error_bloch(bloch_list, pulse_info_list, fidelity_list,
    #                           deltas, epsilons, name=f"Ensemble Evolution of {target_name}",
    #                           save_path=vid, phase_only=True, Omega=Omega)

    animate_multi_error_bloch(bloch_list, pulse_info_list, fidelity_list,
                              deltas, epsilons, name="",
                              save_path=vid, phase_only=True, Omega=Omega)
    return vid, [vid]


# -------------------------------------------------------------------- main
def _predownload_all() -> None:
    if args.hf_weights:
        for k in MODELS:
            _weight_path(k, True)


with gr.Blocks() as demo:
    gr.Markdown("## Composite Pulse for Universal Gates")
    demo.load(lambda: _predownload_all(), outputs=[])
    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(list(MODELS.keys()), value="400 length", label="Model")
            x_in = gr.Number(1.0, label="x-component")
            y_in = gr.Number(0.0, label="y-component")
            z_in = gr.Number(0.0, label="z-component")
            th = gr.Slider(0.0, 2.0, value=0.5, step=0.01, label="Theta (π units)")
            Omega_s = gr.Slider(20, 100, value=80, step=1, label="Omega (2π MHz)")
            cutoff_s = gr.Slider(100, 10000, value=3000, step=100, label="Low-pass cutoff")
            btn1 = gr.Button("Compute Parameters")
            btn2 = gr.Button("Fidelity vs Std")
            btn3 = gr.Button("Evolution Video")
        with gr.Column(scale=2):
            df_out = gr.Dataframe(label="Original Pulse CSV"); csv_dl = gr.File(label="Download Original")
            df_s_out = gr.Dataframe(label="Smoothed Pulse CSV"); csv_s_dl = gr.File(label="Download Smoothed")
            cont_g = gr.Gallery(label="Contour"); param_g = gr.Gallery(label="Param Plot")
            fid_g = gr.Gallery(label="Fidelity vs Std"); vid_out = gr.Video(label="Evolution")
            vid_dl = gr.File(label="Download Video")
    inputs = [model_dd, x_in, y_in, z_in, th, Omega_s, cutoff_s]
    btn1.click(_cb_params, inputs, [df_out, csv_dl, df_s_out, csv_s_dl])
    btn1.click(_cb_contour, inputs, [cont_g])
    btn1.click(_cb_paramplot, inputs, [param_g])
    btn2.click(_cb_fidelity, inputs, [fid_g])
    btn3.click(_cb_evolution, [model_dd, x_in, y_in, z_in, th, Omega_s], [vid_out, vid_dl])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-weights", type=int, default=1,
                        help="1 = pull weights from HuggingFace; 0 = use demo_universal/weight/*.pt")
    args = parser.parse_args()
    demo.launch(share=True)
