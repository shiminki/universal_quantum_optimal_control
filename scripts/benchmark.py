"""Compare checkpoints by pulse duration vs. fidelity.

Each entry is a label:config:checkpoint triple. For every entry the script
measures pulse duration statistics and computes entanglement fidelity.
Results are printed as a table and saved as a Matplotlib figure.


Example usage:
python scripts/benchmark.py \
        --entry "len50:configs/transformer_len50.yaml:checkpoint/len50.pt" \
        --entry "len70:configs/transformer_len70.yaml:checkpoint/len70.pt" \
        --entry "len100:configs/transformer_len100.yaml:checkpoint/len100.pt" \
        --entry "len200:configs/transformer_len200.yaml:checkpoint/len200.pt" \
        --entry "len400:configs/transformer_len400.yaml:checkpoint/len400.pt" \
        --out figures/benchmark.png

python scripts/benchmark.py \
        --entry "len35:configs/old/transformer_len35.yaml:weights/len35.pt" \
        --entry "len50:configs/old/transformer_len50.yaml:weights/len50.pt" \
        --entry "len100:configs/old/transformer_len100.yaml:demo_universal/weight/length_100.pt" \
        --entry "len200:configs/old/transformer_len200.yaml:weights/len200.pt" \
        --entry "len400:configs/old/transformer_len400.yaml:demo_universal/weight/length_400.pt" \
        --out figures/benchmark_specific_target.png \
        --specific-target
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Allow running without `pip install -e .` (e.g. Google Colab)
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import fidelity
from uqoc.models import build_model
from uqoc.pipeline import Pipeline
from uqoc.propagator import batched_unitary_generator
from uqoc.utils import default_device, set_seed


def _evaluate_entry(label: str, cfg_path: Path, ckpt_path: Path,
                    eval_size: int, monte_carlo: int, device: str,
                    warmup: int = 3, specific_target: bool = False) -> dict:
    cfg = load_config(cfg_path)
    model = build_model(cfg.model)
    pipeline = Pipeline(model, weight_path=ckpt_path, device=device)

    # Build dataset based on mode
    if specific_target:
        rot_vec, U_target = build_SU2_dataset(0, random=False, specific_target=True, device=device)
        actual_size = 5  # Always 5 specific gates
    else:
        rot_vec, U_target = build_SU2_dataset(eval_size, random=False, specific_target=False, device=device)
        actual_size = eval_size

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = pipeline(rot_vec)
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        pulses = pipeline(rot_vec) # (B, L, 2) where last dim is [phi, tau]
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Extract durations (tau)
    _, tau = pulses.unbind(dim=-1) # each (B, L)

    # Calculate duration per sequence in units of pi
    durations_pi = tau.sum(dim=-1) / math.pi 
    avg_duration = durations_pi.mean().item() 
    std_duration = durations_pi.std().item()

    # Evaluate fidelity under the hardest curriculum stage
    hardest = cfg.training.curriculum[-1]
    sampler = make_sampler(cfg.training.error_sampler, hardest)
    
    # Memory-safe interleave: ensure MC * actual_size isn't too explosive
    pulses_mc = pulses.repeat_interleave(monte_carlo, dim=0)
    U_mc = U_target.repeat_interleave(monte_carlo, dim=0)
    err = sampler(monte_carlo * actual_size).to(device)
    
    with torch.no_grad():
        U_out = batched_unitary_generator(pulses_mc, err)
        F_all = fidelity(U_out, U_mc, model.num_qubits)

    # Compute statistics
    q = torch.tensor([0.05, 0.95])
    quartiles = torch.quantile(F_all, q)
    q1, q3 = quartiles[0], quartiles[1]
    iqr_half = (q3 - q1) / 2

    result = {
        "label": label,
        "avg_pulse_duration": avg_duration,
        "std_pulse_duration": std_duration,
        "F_mean": F_all.mean().item(),
        "F_min": F_all.min().item(),
        "F_max": F_all.max().item(),
        "F_iqr_half": iqr_half.item(),
        "F_std": F_all.std().item(),
    }
    
    # If specific_target mode, extract individual gate fidelities
    if specific_target:
        # Reshape F_all to (5 gates, monte_carlo samples)
        F_per_gate = F_all.reshape(5, monte_carlo)
        
        # Compute mean fidelity for each gate
        result["F_X_pi"] = F_per_gate[0].mean().item()
        result["F_X_pi_2"] = F_per_gate[1].mean().item()
        result["F_X_pi_4"] = F_per_gate[2].mean().item()
        result["F_H"] = F_per_gate[3].mean().item()
        result["F_T"] = F_per_gate[4].mean().item()

    return result


def _plot(results: list[dict], out: Path, specific_target: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    dur_avg = [r["avg_pulse_duration"] for r in results]
    dur_std = [r["std_pulse_duration"] for r in results]
    f_mean = [r["F_mean"] for r in results]
    f_std = [r["F_std"] for r in results]
    labels = [r["label"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    
    if not specific_target:
        # Use errorbar to show both Pulse Duration Std (x) and Fidelity Std (y)
        ax.errorbar(
            dur_avg, f_mean, 
            xerr=dur_std, yerr=f_std, 
            fmt="o-", capsize=4, elinewidth=1, 
            label="Mean Fidelity", color="steelblue", zorder=4
        )
        
        # Optional: Fill between for a smoother look at fidelity variance
        std_lower = [m - s for m, s in zip(f_mean, f_std)]
        std_upper = [m + s for m, s in zip(f_mean, f_std)]
        ax.fill_between(dur_avg, std_lower, std_upper, alpha=0.15, color="steelblue", zorder=2)
        
        # Add legend
        ax.legend(loc="lower right")

    else:
        # Plot each specific target gate as a separate series
        target_gates = [
            ("F_X_pi", "R_x(π)", "steelblue"),
            ("F_X_pi_2", "R_x(π/2)", "orange"),
            ("F_X_pi_4", "R_x(π/4)", "green"),
            ("F_H", "H", "red"),
            ("F_T", "T", "purple")
        ]
        
        for key, gate_name, color in target_gates:
            f_gate = [r[key] for r in results]
            ax.plot(dur_avg, f_gate, 'o-', label=gate_name, color=color, 
                   markersize=8, linewidth=2, zorder=5)
        
        ax.legend(loc="lower right", fontsize=10)
    
    ax.set_xlabel("Average Pulse Duration (π)", fontsize=12)
    ax.set_ylabel("Gate Fidelity", fontsize=12)
    
    if specific_target:
        ax.set_title("Pulse Duration vs. Fidelity (Specific Target Gates)", fontsize=13)
    else:
        ax.set_title("Pulse Duration vs. Fidelity Performance", fontsize=13)
    
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
        
        if args.specific_target:
            print(f"    R_x(π):   {r['F_X_pi']:.4f}")
            print(f"    R_x(π/2): {r['F_X_pi_2']:.4f}")
            print(f"    R_x(π/4): {r['F_X_pi_4']:.4f}")
            print(f"    H:        {r['F_H']:.4f}")
            print(f"    T:        {r['F_T']:.4f}")

    print("\n" + "="*70)
    if args.specific_target:
        print(f"{'Label':<12} {'Avg Dur (π)':>12} {'F_mean':>10} {'R_x(π)':>10} {'R_x(π/2)':>10} {'R_x(π/4)':>10} {'H':>10} {'T':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['label']:<12} {r['avg_pulse_duration']:>12.2f} "
                  f"{r['F_mean']:>10.4f} {r['F_X_pi']:>10.4f} {r['F_X_pi_2']:>10.4f} {r['F_X_pi_4']:>10.4f} {r['F_H']:>10.4f} {r['F_T']:>10.4f}")
    else:
        print(f"{'Label':<12} {'Avg Dur (π)':>12} {'Std Dur':>10} {'F_mean':>10} {'F_std':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['label']:<12} {r['avg_pulse_duration']:>12.2f} {r['std_pulse_duration']:>10.3f} "
                  f"{r['F_mean']:>10.4f} {r['F_std']:>10.4f}")

    _plot(results, args.out, args.specific_target)


if __name__ == "__main__":
    main()