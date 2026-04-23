import math

import pytest
import torch

from uqoc.quantum import (
    euler_yxy,
    fidelity,
    paulis,
    rotation_unitary,
    rotation_vector_to_unitary,
    score_sequence_from_yxy,
    to_real_vector,
)


def test_paulis_algebra():
    p = paulis("cpu").to(torch.cdouble)
    I, X, Y, Z = p[0], p[1], p[2], p[3]
    assert torch.allclose(X @ Y, 1j * Z, atol=1e-12)
    assert torch.allclose(Y @ Z, 1j * X, atol=1e-12)
    assert torch.allclose(Z @ X, 1j * Y, atol=1e-12)
    assert torch.allclose(X @ X, I, atol=1e-12)


def test_rotation_unitary_is_unitary():
    n = torch.tensor([1.0, 0.0, 0.0])
    U = rotation_unitary(n, torch.tensor(math.pi / 2))
    UU = U.conj().transpose(-1, -2) @ U
    assert torch.allclose(UU, torch.eye(2, dtype=U.dtype), atol=1e-5)


def test_rotation_x_half_matches_matrix_exp():
    n = torch.tensor([1.0, 0.0, 0.0])
    theta = torch.tensor(math.pi / 3)
    U_closed = rotation_unitary(n, theta).to(torch.cdouble)
    p = paulis("cpu").to(torch.cdouble)
    U_expm = torch.matrix_exp(-1j * theta / 2 * p[1])
    assert torch.allclose(U_closed, U_expm, atol=1e-10)


def test_euler_yxy_roundtrip():
    # random batch of rotation vectors; decompose then recompose.
    torch.manual_seed(0)
    B = 8
    n = torch.randn(B, 3)
    n = n / n.norm(dim=1, keepdim=True)
    theta = torch.rand(B) * math.pi
    rot_vec = torch.cat([n, theta.unsqueeze(1)], dim=1)
    euler = euler_yxy(rot_vec)
    alpha, beta, gamma = euler.unbind(-1)
    y = torch.tensor([0.0, 1.0, 0.0])
    x = torch.tensor([1.0, 0.0, 0.0])
    U_recon = (rotation_unitary(y.expand(B, 3), alpha)
               @ rotation_unitary(x.expand(B, 3), beta)
               @ rotation_unitary(y.expand(B, 3), gamma))
    U_true = rotation_vector_to_unitary(rot_vec)
    # SU(2) is a double-cover of SO(3); ±U are the same rotation. Compare up to global phase.
    global_phase = (U_recon[:, 0, 0] / U_true[:, 0, 0]).abs()
    assert torch.allclose(global_phase, torch.ones_like(global_phase), atol=1e-4)


def test_fidelity_identity_is_one():
    U = rotation_vector_to_unitary(torch.tensor([[1.0, 0.0, 0.0, 0.7]]))
    assert torch.allclose(fidelity(U, U), torch.ones(1), atol=1e-6)


def test_to_real_vector_roundtrip():
    torch.manual_seed(0)
    U = torch.randn(4, 2, 2, dtype=torch.cfloat)
    v = to_real_vector(U)
    assert v.shape == (4, 8)
    # real/imag are interleaved — first two entries match real, imag of U[b, 0, 0].
    assert torch.allclose(v[0, 0], U[0, 0, 0].real)
    assert torch.allclose(v[0, 1], U[0, 0, 0].imag)


def test_score_sequence_shape():
    torch.manual_seed(0)
    euler = torch.randn(4, 3)
    seq = score_sequence_from_yxy(euler)
    assert seq.shape == (4, 9, 2, 2)
    # each 2x2 block must be unitary
    UU = seq.conj().transpose(-1, -2) @ seq
    I = torch.eye(2, dtype=UU.dtype).expand_as(UU)
    assert torch.allclose(UU, I, atol=1e-4)
