import torch

from uqoc.models import MODEL_REGISTRY, build_model


def _tiny_transformer():
    return build_model({
        "type": "transformer", "num_qubits": 1,
        "pulse_space": {"phi": (-3.15, 3.15), "tau": (0.1, 0.5)},
        "max_pulses": 8, "d_model": 32, "n_layers": 2, "n_heads": 4, "dropout": 0.1,
    })


def _tiny_deep_nn():
    return build_model({
        "type": "deep_nn",
        "pulse_space": {"phi": (-3.15, 3.15), "tau": (0.01, 0.5)},
        "num_pulses": 8,
    })


def test_registry_has_both():
    assert {"transformer", "deep_nn"} <= set(MODEL_REGISTRY)


def test_transformer_forward_shape():
    m = _tiny_transformer()
    rot_vec = torch.tensor([[1.0, 0.0, 0.0, 1.5]])
    pulses = m(rot_vec)
    assert pulses.shape == (1, 8, 2)


def test_deep_nn_forward_shape():
    m = _tiny_deep_nn()
    rot_vec = torch.tensor([[1.0, 0.0, 0.0, 1.5]])
    pulses = m(rot_vec)
    assert pulses.shape == (1, 8, 2)


def test_transformer_tau_nonneg_and_phi_in_range():
    m = _tiny_transformer()
    rot_vec = torch.randn(4, 4)
    rot_vec[:, :3] = rot_vec[:, :3] / rot_vec[:, :3].norm(dim=1, keepdim=True)
    pulses = m(rot_vec)
    phi, tau = pulses[..., 0], pulses[..., 1]
    assert (tau >= 0).all()
    assert (phi >= -torch.pi - 1e-4).all() and (phi <= torch.pi + 1e-4).all()


def test_build_model_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        build_model({"type": "does_not_exist"})
