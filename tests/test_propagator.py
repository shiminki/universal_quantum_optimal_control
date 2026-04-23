import math

import torch

from uqoc.propagator import batched_unitary_generator
from uqoc.quantum import paulis


def test_single_pulse_matches_closed_form():
    # One pulse (phi=0, tau=pi) with no errors → exp(-i pi/2 X) = -i X
    pulses = torch.tensor([[[0.0, math.pi]]])
    error = torch.zeros(2, 1)
    U = batched_unitary_generator(pulses, error)
    X = paulis("cpu").to(torch.cfloat)[1]
    expected = -1j * X
    assert torch.allclose(U[0], expected, atol=1e-5)


def test_zero_tau_identity():
    pulses = torch.zeros(3, 5, 2)       # tau=0 everywhere
    error = torch.zeros(2, 3)
    U = batched_unitary_generator(pulses, error)
    I = torch.eye(2, dtype=U.dtype).expand(3, 2, 2)
    assert torch.allclose(U, I, atol=1e-6)


def test_composite_is_product_left_to_right():
    # U_total should equal U_L @ ... @ U_1. Verify with two non-trivial pulses.
    pulses = torch.tensor([[[0.0, 0.4], [math.pi / 2, 0.3]]])
    error = torch.zeros(2, 1)
    U_total = batched_unitary_generator(pulses, error)[0]
    U1 = batched_unitary_generator(pulses[:, :1, :], error)[0]
    U2 = batched_unitary_generator(pulses[:, 1:, :], error)[0]
    assert torch.allclose(U_total, U2 @ U1, atol=1e-5)


def test_propagator_is_unitary_under_error():
    pulses = torch.randn(2, 8, 2)
    pulses[:, :, 1] = pulses[:, :, 1].abs()  # tau >= 0
    error = torch.stack([torch.randn(2), 0.05 * torch.randn(2)])
    U = batched_unitary_generator(pulses, error)
    UU = U.conj().transpose(-1, -2) @ U
    I = torch.eye(2, dtype=UU.dtype).expand_as(UU)
    assert torch.allclose(UU, I, atol=1e-4)
