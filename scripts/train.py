"""Train any registered controller from a YAML config.

Usage:
    python scripts/train.py --config configs/transformer_len100.yaml \\
                            --save-dir weights/len100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running without `pip install -e .` (e.g. Google Colab)
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from uqoc.bandwidth import make_bandwidth
from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import make_loss
from uqoc.models import build_model
from uqoc.propagator import batched_unitary_generator
from uqoc.trainer import Trainer
from uqoc.utils import default_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = default_device(args.device)
    print(f"[train] config={args.config}  device={device}  Ω={cfg.physics.omega_mhz} MHz"
          f"  bandwidth={cfg.bandwidth.mode}@{cfg.bandwidth.cutoff_mhz} MHz")

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

    def sampler_factory(params):
        return make_sampler(cfg.training.error_sampler, params,
                            device=device, omega_mhz=cfg.physics.omega_mhz)

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
