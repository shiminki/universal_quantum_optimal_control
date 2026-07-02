"""Compare checkpoints by pulse duration vs. fidelity.

Each entry is a label:config:checkpoint triple. For every entry the script
applies the config's bandwidth transform (if any), measures pulse duration
statistics, entanglement fidelity, and rotation-angle error under the hardest
curriculum stage. Results are printed as a table and saved as a
publication-ready figure (mean line, mean ± std shaded band, no titles).


Example usage:
python scripts/benchmark.py \
        --entry "len50:configs/transformer_len50.yaml:checkpoint/len50.pt" \
        --entry "len70:configs/transformer_len70.yaml:checkpoint/len70.pt" \
        --entry "len100:configs/transformer_len100.yaml:checkpoint/len100.pt" \
        --entry "len200:configs/transformer_len200.yaml:checkpoint/len200.pt" \
        --entry "len400:configs/transformer_len400.yaml:checkpoint/len400.pt" \
        --out figures/benchmark.png

python scripts/benchmark.py \
        --entry "len35:configs/transformer_len35.yaml:weights/len35.pt" \
        --entry "len50:configs/transformer_len50.yaml:weights/len50.pt" \
        --entry "len100:configs/transformer_len100.yaml:demo_universal/weight/length_100.pt" \
        --entry "len200:configs/transformer_len200.yaml:weights/len200.pt" \
        --entry "len400:configs/transformer_len400.yaml:demo_universal/weight/length_400.pt" \
        --out figures/benchmark_specific_target.png \
        --specific-target
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running without `pip install -e .` (e.g. Google Colab)
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from uqoc.bandwidth import make_bandwidth
from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import rotation_angle_error
from uqoc.utils import default_device, set_seed

# Validated categorical palette (fixed slot order — see dataviz palette).
_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
_SERIES_MARKERS = ["o", "s", "^", "D", "v"]
_GRID_KW = dict(alpha=0.25, linewidth=0.6)


def _evaluate_entry(label: str, cfg_path: Path, ckpt_path: Path,
                    eval_size: int, monte_carlo: int, device: str,
                    specific_target: bool = False) -> dict:
    cfg = load_config(cfg_path)
    model = build_model(cfg.model)
    pipeline = Pipeline(model, weight_path=ckpt_path, device=device)
    pulse_transform, _ = make_bandwidth(
        cfg.bandwidth.mode, cfg.bandwidth.cutoff_mhz, cfg.physics.omega_mhz,
        max_radius=cfg.bandwidth.max_radius, oversample=cfg.bandwidth.oversample,
    )

    # Build dataset based on mode
    if specific_target:
        rot_vec, U_target = build_SU2_dataset(0, random=False, specific_target=True, device=device)
        actual_size = 5  # Always 5 specific gates
    else:
        rot_vec, U_target = build_SU2_dataset(eval_size, random=False, specific_target=False, device=device)
        actual_size = eval_size

    with torch.no_grad():
        pulses = pipeline(rot_vec)  # (B, L, 2) where last dim is [phi, tau]
        if pulse_transform is not None:
            pulses = pulse_transform(pulses)

    # Extract durations (tau)
    _, tau = pulses.unbind(dim=-1)  # each (B, L)

    # Calculate duration per sequence in units of pi
    durations_pi = tau.sum(dim=-1) / math.pi
    avg_duration = durations_pi.mean().item()
    std_duration = durations_pi.std().item()

    # Evaluate fidelity under the hardest curriculum stage
    hardest = cfg.training.curriculum[-1]
    sampler = make_sampler(cfg.training.error_sampler, hardest,
                           device=device, omega_mhz=cfg.physics.omega_mhz)

    pulses_mc = pulses.repeat_interleave(monte_carlo, dim=0)
    U_mc = U_target.repeat_interleave(monte_carlo, dim=0)
    err = sampler(monte_carlo * actual_size).to(device)

    with torch.no_grad():
        U_out = batched_unitary_generator(pulses_mc, err)
        F_all = fidelity(U_out, U_mc, model.num_qubits)
        theta_err = rotation_angle_error(U_out, U_mc)

    result = {
        "label": label,
        "avg_pulse_duration": avg_duration,
        "std_pulse_duration": std_duration,
        "F_mean": F_all.mean().item(),
        "F_std": F_all.std().item(),
        "theta_rmse": theta_err.pow(2).mean().sqrt().item(),
    }

    # If specific_target mode, extract per-gate fidelity statistics
    if specific_target:
        F_per_gate = F_all.reshape(5, monte_carlo)
        for i, key in enumerate(["F_X_pi", "F_X_pi_2", "F_X_pi_4", "F_H", "F_T"]):
            result[key] = F_per_gate[i].mean().item()
            result[key + "_std"] = F_per_gate[i].std().item()

    return result


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, **_GRID_KW)
    ax.tick_params(labelsize=11)


def _plot(results: list[dict], out: Path, specific_target: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort([r["avg_pulse_duration"] for r in results])
    results = [results[i] for i in order]
    dur = np.array([r["avg_pulse_duration"] for r in results])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    if not specific_target:
        f_mean = np.array([r["F_mean"] for r in results])
        f_std = np.array([r["F_std"] for r in results])
        ax.fill_between(dur, f_mean - f_std, np.minimum(f_mean + f_std, 1.0),
                        color=_SERIES_COLORS[0], alpha=0.18, linewidth=0,
                        label=r"mean $\pm$ 1 s.d.")
        ax.plot(dur, f_mean, marker="o", markersize=6, linewidth=2,
                color=_SERIES_COLORS[0], label="mean fidelity")
        ax.legend(loc="lower right", frameon=False, fontsize=11)
    else:
        target_gates = [
            ("F_X_pi", r"$R_x(\pi)$"),
            ("F_X_pi_2", r"$R_x(\pi/2)$"),
            ("F_X_pi_4", r"$R_x(\pi/4)$"),
            ("F_H", r"$H$"),
            ("F_T", r"$T$"),
        ]
        for i, (key, gate_name) in enumerate(target_gates):
            f_gate = np.array([r[key] for r in results])
            f_gstd = np.array([r[key + "_std"] for r in results])
            ax.fill_between(dur, f_gate - f_gstd, np.minimum(f_gate + f_gstd, 1.0),
                            color=_SERIES_COLORS[i], alpha=0.12, linewidth=0)
            ax.plot(dur, f_gate, marker=_SERIES_MARKERS[i], markersize=6,
                    linewidth=2, color=_SERIES_COLORS[i], label=gate_name,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)

    ax.set_xlabel(r"Pulse duration ($\pi$)", fontsize=13)
    ax.set_ylabel("Average fidelity", fontsize=13)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[benchmark] figure saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark: Pulse Duration vs. Fidelity")
    parser.add_argument("--entry", action="append", required=True, metavar="LABEL:CONFIG:CKPT")
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--monte-carlo", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--specific-target", action="store_true",
                        help="Use 5 specific target gates instead of random sampling")
    parser.add_argument("--out", type=Path, default=Path("figures/benchmark_duration.png"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = default_device(args.device)

    if args.specific_target:
        print(f"[benchmark] MODE: Specific target gates (5 gates)")
        print(f"[benchmark] device={device}  MC={args.monte_carlo}")
    else:
        print(f"[benchmark] MODE: Random eval dataset")
        print(f"[benchmark] device={device}  eval_size={args.eval_size}  MC={args.monte_carlo}")

    results = []
    for raw in args.entry:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            parser.error(f"--entry must be LABEL:CONFIG:CKPT, got: {raw!r}")
        label, cfg_str, ckpt_str = parts

        print(f"\n[benchmark] evaluating {label!r} ...")
        r = _evaluate_entry(label, Path(cfg_str), Path(ckpt_str),
                            args.eval_size, args.monte_carlo, device,
                            specific_target=args.specific_target)
        results.append(r)

        print(f"  Duration: {r['avg_pulse_duration']:.2f} ± {r['std_pulse_duration']:.2f} π")
        print(f"  Fidelity: {r['F_mean']:.4f} ± {r['F_std']:.4f}")
        print(f"  θ RMSE:   {r['theta_rmse']:.4f} rad")

        if args.specific_target:
            print(f"    R_x(π):   {r['F_X_pi']:.4f} ± {r['F_X_pi_std']:.4f}")
            print(f"    R_x(π/2): {r['F_X_pi_2']:.4f} ± {r['F_X_pi_2_std']:.4f}")
            print(f"    R_x(π/4): {r['F_X_pi_4']:.4f} ± {r['F_X_pi_4_std']:.4f}")
            print(f"    H:        {r['F_H']:.4f} ± {r['F_H_std']:.4f}")
            print(f"    T:        {r['F_T']:.4f} ± {r['F_T_std']:.4f}")

    print("\n" + "=" * 84)
    if args.specific_target:
        print(f"{'Label':<12} {'Avg Dur (π)':>12} {'F_mean':>9} {'θ RMSE':>9} "
              f"{'R_x(π)':>9} {'R_x(π/2)':>9} {'R_x(π/4)':>9} {'H':>9} {'T':>9}")
        print("-" * 84)
        for r in results:
            print(f"{r['label']:<12} {r['avg_pulse_duration']:>12.2f} "
                  f"{r['F_mean']:>9.4f} {r['theta_rmse']:>9.4f} "
                  f"{r['F_X_pi']:>9.4f} {r['F_X_pi_2']:>9.4f} {r['F_X_pi_4']:>9.4f} "
                  f"{r['F_H']:>9.4f} {r['F_T']:>9.4f}")
    else:
        print(f"{'Label':<12} {'Avg Dur (π)':>12} {'Std Dur':>10} {'F_mean':>10} "
              f"{'F_std':>10} {'θ RMSE':>10}")
        print("-" * 84)
        for r in results:
            print(f"{r['label']:<12} {r['avg_pulse_duration']:>12.2f} {r['std_pulse_duration']:>10.3f} "
                  f"{r['F_mean']:>10.4f} {r['F_std']:>10.4f} {r['theta_rmse']:>10.4f}")

    _plot(results, args.out, args.specific_target)


if __name__ == "__main__":
    main()
