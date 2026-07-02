"""Train any registered controller with detuning drawn uniformly from a union of intervals.

Instead of the Gaussian δ/Ω ~ N(0,1) used in train.py, this script samples δ/Ω
uniformly from Union(intervals), where each interval contributes weight proportional
to its length.

Usage:
    python scripts/train_with_peaks.py --config configs/transformer_len100.yaml \\
                                       --save-dir weights/peaks \\
                                       --intervals -3.6 -3 -1.3 -0.7 0.7 1.3 3 3.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, List, Tuple

import torch

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from uqoc.bandwidth import make_bandwidth
from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.fidelity import make_loss
from uqoc.models import build_model
from uqoc.propagator import batched_unitary_generator
from uqoc.trainer import Trainer
from uqoc.utils import default_device, set_seed

_DEFAULT_INTERVALS: List[float] = [-3.6, -3, -1.3, -0.7, 0.7, 1.3, 3, 3.6]


def _parse_intervals(flat: List[float]) -> List[Tuple[float, float]]:
    if len(flat) % 2 != 0:
        raise ValueError(f"--intervals requires an even number of values, got {len(flat)}")
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    for lo, hi in pairs:
        if lo >= hi:
            raise ValueError(f"Interval [{lo}, {hi}] is empty or reversed")
    return pairs


def make_peaks_sampler(
    intervals: List[Tuple[float, float]],
    epsilon_std: float = 0.0,
    device: torch.device | str | None = None,
) -> Callable[[int], torch.Tensor]:
    """Return a sampler that draws δ/Ω uniformly from Union(intervals)."""
    lows = torch.tensor([lo for lo, _ in intervals], device=device)
    widths = torch.tensor([hi - lo for lo, hi in intervals], device=device)
    weights = widths / widths.sum()

    def sampler(batch_size: int) -> torch.Tensor:
        idx = torch.multinomial(weights, batch_size, replacement=True)
        delta = lows[idx] + torch.rand(batch_size, device=device) * widths[idx]
        epsilon = (torch.randn(batch_size, device=device) * epsilon_std if epsilon_std > 0
                   else torch.zeros(batch_size, device=device))
        return torch.stack([delta, epsilon])

    return sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto if omitted)")
    parser.add_argument(
        "--intervals",
        nargs="+",
        type=float,
        default=_DEFAULT_INTERVALS,
        metavar="VAL",
        help="Flat list of interval endpoints (lo hi lo hi …). "
             f"Default: {_DEFAULT_INTERVALS}",
    )
    parser.add_argument(
        "--epsilon-std",
        type=float,
        default=0.0,
        help="Std-dev of pulse-length error ε (0 = ORE only, default: 0.0)",
    )
    args = parser.parse_args()

    intervals = _parse_intervals(args.intervals)
    print(f"[train_with_peaks] intervals = {intervals}")
    print(f"[train_with_peaks] epsilon_std = {args.epsilon_std}")

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = default_device(args.device)
    print(f"[train_with_peaks] config={args.config}  device={device}")

    model = build_model(cfg.model)
    loss_fn = make_loss(cfg.training.loss, cfg.training.loss_params)
    pulse_transform, pulse_penalty = make_bandwidth(
        cfg.bandwidth.mode, cfg.bandwidth.cutoff_mhz, cfg.physics.omega_mhz,
        max_radius=cfg.bandwidth.max_radius, oversample=cfg.bandwidth.oversample,
    )

    trainer = Trainer(
        model,
        unitary_generator=batched_unitary_generator,
        loss_fn=loss_fn,
        monte_carlo=cfg.training.monte_carlo,
        learning_rate=cfg.training.learning_rate,
        grad_clip=cfg.training.grad_clip,
        device=device,
        pulse_transform=pulse_transform,
        pulse_penalty=pulse_penalty,
        penalty_weight=cfg.bandwidth.penalty_weight,
    )

    train_rot, train_U = build_SU2_dataset(cfg.dataset.train_size, random=cfg.dataset.random)
    eval_rot, eval_U = build_SU2_dataset(cfg.dataset.eval_size, random=cfg.dataset.random)

    peaks_sampler = make_peaks_sampler(intervals, epsilon_std=args.epsilon_std, device=device)

    # curriculum params are ignored for detuning; only the epoch structure is used
    def sampler_factory(_params: dict) -> Callable[[int], torch.Tensor]:
        return peaks_sampler

    trainer.train(
        train_rot_vec=train_rot,
        train_U=train_U,
        eval_rot_vec=eval_rot,
        eval_U=eval_U,
        curriculum=cfg.training.curriculum,
        error_sampler_factory=sampler_factory,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
