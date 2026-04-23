"""True GRAPE: gradient ascent directly on pulse parameters.

Unlike `DeepNNController`, which parameterises pulses as the output of a neural
network conditioned on the target, this baseline optimises a single
(num_pulses, 2) tensor of raw (phi, tau) values for **one** target gate at a
time. No generalisation across SU(2); this is the textbook GRAPE setup.

Usage:
    python -m baselines.true_grape --target X --num-pulses 100 --epochs 2000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from uqoc.errors import make_sampler
from uqoc.fidelity import LOSS_REGISTRY, fidelity
from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import rotation_vector_to_unitary


TARGETS = {
    "X": (1.0, 0.0, 0.0, math.pi),
    "X_half": (1.0, 0.0, 0.0, math.pi / 2),
    "Y": (0.0, 1.0, 0.0, math.pi),
    "Z_quarter": (0.0, 0.0, 1.0, math.pi / 4),
    "Hadamard": (1.0, 0.0, 1.0, math.pi),
}


class TrueGRAPE(nn.Module):
    """A single learnable pulse sequence (phi, tau). Optimised directly, no NN."""

    def __init__(self, num_pulses: int,
                 phi_range: tuple[float, float] = (-math.pi, math.pi),
                 tau_range: tuple[float, float] = (0.01, 0.5)) -> None:
        super().__init__()
        # Unconstrained logits passed through sigmoid → (low, high)
        self.phi_logits = nn.Parameter(torch.randn(num_pulses))
        self.tau_logits = nn.Parameter(torch.randn(num_pulses))
        self.phi_range = phi_range
        self.tau_range = tau_range

    def forward(self) -> torch.Tensor:
        phi = self.phi_range[0] + (self.phi_range[1] - self.phi_range[0]) * self.phi_logits.sigmoid()
        tau = self.tau_range[0] + (self.tau_range[1] - self.tau_range[0]) * self.tau_logits.sigmoid()
        return torch.stack([phi, tau], dim=-1)     # (L, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), default="X")
    parser.add_argument("--num-pulses", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--monte-carlo", type=int, default=256)
    parser.add_argument("--delta-std", type=float, default=0.4)
    parser.add_argument("--epsilon-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save", type=Path, default=Path("baselines/true_grape.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    nx, ny, nz, theta = TARGETS[args.target]
    rot_vec = torch.tensor([nx, ny, nz, theta], dtype=torch.float32)
    U_target = rotation_vector_to_unitary(rot_vec).unsqueeze(0)

    model = TrueGRAPE(args.num_pulses)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = make_sampler("ore_ple", {"delta_std": args.delta_std, "epsilon_std": args.epsilon_std})
    loss_fn = LOSS_REGISTRY["infidelity"]

    best = 0.0
    with tqdm(range(args.epochs), desc=f"GRAPE[{args.target}]") as pbar:
        for _ in pbar:
            pulse = model().unsqueeze(0).repeat_interleave(args.monte_carlo, 0)
            U_mc = U_target.repeat_interleave(args.monte_carlo, 0)
            err = sampler(args.monte_carlo)
            U_out = batched_unitary_generator(pulse, err)
            loss = loss_fn(U_out, U_mc, 1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            F = fidelity(U_out, U_mc, 1).mean().item()
            best = max(best, F)
            pbar.set_postfix(loss=float(loss), fid=F, best=best)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pulse": model().detach(), "target": args.target,
                "delta_std": args.delta_std, "epsilon_std": args.epsilon_std,
                "fid": best}, args.save)
    print(f"saved → {args.save}  best F = {best:.6f}")


if __name__ == "__main__":
    main()
