"""Load a checkpoint and compute mean fidelity on a freshly sampled eval set.

Usage:
    python scripts/evaluate.py --config configs/transformer_len100.yaml \\
                               --checkpoint weights/len100/err_delta_std1.000_epsilon_std0.050.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.utils import default_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--monte-carlo", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=1024)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = default_device(args.device)

    model = build_model(cfg.model)
    pipeline = Pipeline(model, weight_path=args.checkpoint, device=device)

    rot_vec, U_target = build_SU2_dataset(args.eval_size, random=True)
    rot_vec, U_target = rot_vec.to(device), U_target.to(device)

    pulses = pipeline(rot_vec)
    for params in cfg.training.curriculum:
        sampler = make_sampler(cfg.training.error_sampler, params)
        pulses_mc = pulses.repeat_interleave(args.monte_carlo, dim=0)
        U_mc = U_target.repeat_interleave(args.monte_carlo, dim=0)
        err = sampler(args.monte_carlo * rot_vec.shape[0]).to(device)
        with torch.no_grad():
            U_out = batched_unitary_generator(pulses_mc, err)
            fidelity_list = fidelity(U_out, U_mc, model.num_qubits)
            F = fidelity_list.mean().item()
            F_min = fidelity_list.min().item()
        print(f"  {params} → F = {F:.6f}, F_min = {F_min:.6f}")


if __name__ == "__main__":
    main()
