import torch

from uqoc.dataset import build_SU2_dataset
from uqoc.errors import make_sampler
from uqoc.fidelity import LOSS_REGISTRY
from uqoc.models import build_model
from uqoc.propagator import batched_unitary_generator
from uqoc.trainer import Trainer


def test_loss_decreases_in_20_steps():
    torch.manual_seed(0)
    model = build_model({
        "type": "deep_nn",
        "pulse_space": {"phi": (-3.15, 3.15), "tau": (0.1, 0.5)},
        "num_pulses": 16,
    })
    trainer = Trainer(
        model, batched_unitary_generator, LOSS_REGISTRY["infidelity"],
        monte_carlo=16, learning_rate=3e-4, device="cpu",
    )
    rot, U = build_SU2_dataset(16, random=True)
    sampler = make_sampler("ore_ple", {"delta_std": 0.4, "epsilon_std": 0.05})

    losses = [trainer.train_step(rot, U, sampler) for _ in range(20)]
    # average of last 5 < average of first 5 — loss trending down
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
