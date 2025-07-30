import gradio as gr
import torch, math, os, sys, glob, json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

# Ensure project root is on PYTHONPATH so we can import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from train.unitary_single_qubit_gate.unitary_single_qubit_gate import *
from visualize.util import *

# ------------------------------------------------------------------
# Constants & helpers (copied from original Streamlit demo)
# ------------------------------------------------------------------
_I2_CPU       = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU  = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cfloat)
_SIGMA_Y_CPU  = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.cfloat)
_SIGMA_Z_CPU  = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cfloat)
pauli         = [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU]

def Rx(theta):
    return expm(-1j * _SIGMA_X_CPU * theta / 2)

def Ry(theta):
    return expm(-1j * _SIGMA_Y_CPU * theta / 2)

# ------------------------------------------------------------------
# Core inference (this is where the heavy lifting happens)
# ------------------------------------------------------------------
def run_inference(model_len: str, x: float, y: float, z: float, theta_pi: float,
                  do_fid_plot: bool, do_anim: bool, progress=gr.Progress(track_tqdm=True)):
    # [unchanged inference code]
    # returns: md_output, csv_path, pulse_tensor, contour_imgs, param_imgs, std_imgs, video_path
    return md_output, csv_path, pulse_tensor.detach().numpy(), contour_imgs, param_imgs, std_imgs, video_path

# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title='Composite Pulse for Universal Gates (Gradio)') as demo:
    gr.Markdown('# Composite Pulse for Universal Gates')

    with gr.Row():
        # Dropdown without undefined MODEL_META
        model_len   = gr.Dropdown(label='Select model length', choices=['100 length', '400 length'], value='100 length')
        do_fid_plot = gr.Checkbox(label='Run fidelity vs std plot')
        do_anim     = gr.Checkbox(label='Run qubit evolution animation')

    gr.Markdown('### Specify target unitary')
    with gr.Row():
        x_in = gr.Number(value=1.0, label='x‑component')
        y_in = gr.Number(value=0.0, label='y‑component')
        z_in = gr.Number(value=0.0, label='z‑component')
    theta_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.01,
                             label='Theta (units of π)')

    run_btn = gr.Button('Run Inference')

    # ---------- Outputs ----------
    summary_md   = gr.Markdown()
    csv_file_out = gr.File(label='Download pulse CSV')
    df_out       = gr.Dataframe(headers=None)
    contour_out  = gr.Gallery(label='Fidelity Contour', columns=3)
    param_out    = gr.Gallery(label='Pulse Parameter Plot', columns=3)
    std_out      = gr.Gallery(label='Fidelity vs Delta Std', columns=3)
    video_out    = gr.Video()

    run_btn.click(
        run_inference,
        [model_len, x_in, y_in, z_in, theta_slider, do_fid_plot, do_anim],
        [summary_md, csv_file_out, df_out, contour_out, param_out, std_out, video_out]
    )

if __name__ == '__main__':
    demo.launch()
