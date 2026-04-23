"""Train any registered controller from a YAML config.

Usage:
    python scripts/train.py --config configs/transformer_len100.yaml \\
                            --save-dir weights/len100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uqoc.config import load_config
from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import LOSS_REGISTRY
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
    print(f"[train] config={args.config}  device={device}")

    model = build_model(cfg.model)
    loss_fn = LOSS_REGISTRY[cfg.training.loss]

    trainer = Trainer(
        model,
        unitary_generator=batched_unitary_generator,
        loss_fn=loss_fn,
        monte_carlo=cfg.training.monte_carlo,
        learning_rate=cfg.training.learning_rate,
        grad_clip=cfg.training.grad_clip,
        device=device,
    )

    train_rot, train_U = build_SU2_dataset(cfg.dataset.train_size, random=cfg.dataset.random)
    eval_rot, eval_U = build_SU2_dataset(cfg.dataset.eval_size, random=cfg.dataset.random)

    def sampler_factory(params):
        return make_sampler(cfg.training.error_sampler, params)

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
