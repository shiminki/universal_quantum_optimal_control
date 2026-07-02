"""Load a checkpoint and compute fidelity / rotation-angle statistics on a fresh eval set.

Applies the config's bandwidth transform (if any) before propagation, reports
mean ± std fidelity and rotation-angle RMSE per curriculum stage, and with
--hbn also evaluates at the discrete inherent hBN detuning lines
(δ ∈ {−96, −32, +32, +96} MHz).

Usage:
    python scripts/evaluate.py --config configs/transformer_len50_hbn30.yaml \
                               --checkpoint checkpoint/len50_hbn30.pt --hbn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from uqoc.bandwidth import make_bandwidth
from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import HBN_DETUNINGS_MHZ, make_sampler
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import rotation_angle_error
from uqoc.utils import default_device, set_seed


@torch.no_grad()
def _evaluate(pulses: torch.Tensor, U_target: torch.Tensor, sampler, monte_carlo: int,
              num_qubits: int, device) -> tuple[float, float, float]:
    """Returns (F_mean, F_std, theta_rmse) over the MC-expanded batch."""
    B = pulses.shape[0]
    pulses_mc = pulses.repeat_interleave(monte_carlo, dim=0)
    U_mc = U_target.repeat_interleave(monte_carlo, dim=0)
    err = sampler(monte_carlo * B).to(device)
    U_out = batched_unitary_generator(pulses_mc, err)
    F = fidelity(U_out, U_mc, num_qubits)
    theta_err = rotation_angle_error(U_out, U_mc)
    return F.mean().item(), F.std().item(), theta_err.pow(2).mean().sqrt().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--monte-carlo", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--hbn", action="store_true",
                        help="Also evaluate at the discrete hBN detuning lines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = default_device(args.device)

    model = build_model(cfg.model)
    pipeline = Pipeline(model, weight_path=args.checkpoint, device=device)
    pulse_transform, _ = make_bandwidth(
        cfg.bandwidth.mode, cfg.bandwidth.cutoff_mhz, cfg.physics.omega_mhz,
        max_radius=cfg.bandwidth.max_radius, oversample=cfg.bandwidth.oversample,
    )

    rot_vec, U_target = build_SU2_dataset(args.eval_size, random=True)
    rot_vec, U_target = rot_vec.to(device), U_target.to(device)

    pulses = pipeline(rot_vec)
    if pulse_transform is not None:
        with torch.no_grad():
            pulses = pulse_transform(pulses)

    print(f"[evaluate] Ω = {cfg.physics.omega_mhz} MHz, "
          f"bandwidth = {cfg.bandwidth.mode}@{cfg.bandwidth.cutoff_mhz} MHz")
    for params in cfg.training.curriculum:
        sampler = make_sampler(cfg.training.error_sampler, params,
                               device=device, omega_mhz=cfg.physics.omega_mhz)
        F, F_std, theta_rmse = _evaluate(pulses, U_target, sampler, args.monte_carlo,
                                         model.num_qubits, device)
        print(f"  {params} → F = {F:.6f} ± {F_std:.6f}, θ RMSE = {theta_rmse:.5f} rad")

    if args.hbn:
        sampler = make_sampler("hbn_discrete", {}, device=device,
                               omega_mhz=cfg.physics.omega_mhz)
        F, F_std, theta_rmse = _evaluate(pulses, U_target, sampler, args.monte_carlo,
                                         model.num_qubits, device)
        print(f"  hBN lines δ ∈ {list(HBN_DETUNINGS_MHZ)} MHz "
              f"→ F = {F:.6f} ± {F_std:.6f}, θ RMSE = {theta_rmse:.5f} rad")


if __name__ == "__main__":
    main()
