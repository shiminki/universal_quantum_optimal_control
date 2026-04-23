import math

import pytest
import torch

from uqoc.dataset import build_SU2_dataset


def test_random_mode_shape_and_axis_norm():
    rot, U = build_SU2_dataset(64, random=True)
    assert rot.shape == (64, 4)
    assert U.shape == (64, 2, 2)
    axis_norm = rot[:, :3].norm(dim=1)
    assert torch.allclose(axis_norm, torch.ones(64), atol=1e-5)


def test_grid_mode_requires_perfect_square():
    with pytest.raises(ValueError):
        build_SU2_dataset(50, random=False)


def test_unitary_is_unitary():
    rot, U = build_SU2_dataset(16, random=True)
    UU = U.conj().transpose(-1, -2) @ U
    I = torch.eye(2, dtype=UU.dtype).expand_as(UU)
    assert torch.allclose(UU, I, atol=1e-4)


def test_theta_range():
    rot, _ = build_SU2_dataset(256, random=True)
    theta = rot[:, 3]
    assert (theta >= 0).all() and (theta <= 2 * math.pi + 1e-6).all()
