from model.universal_model import UniversalQOCTransformer
from model.universal_model_trainer import UniversalModelTrainer
from train.unitary_single_qubit_gate.universal_single_qubit_SCORE import *
from model.universal_model import UniversalQOCTransformer
import torch
import numpy as np

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)


model_param = {
    "num_qubits": 1,
    "pulse_space": {
        "Delta": (-5, 5), "Omega": (0, 1), "phi": (-torch.pi, torch.pi), "tau": (0, 1.0)
    },
    "max_pulses": 10,
}

model = UniversalQOCTransformer(**model_param)


U_targets = torch.stack([
    _I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU
])


# pulses = model(U_targets)

# print(pulses)
# print(pulses.shape)




# ###########################
# # Test ####################
# ###########################

# # Step 1: Initialize model
# model_param = {
#     "num_qubits": 1,
#     "pulse_space": {
#         "Delta": (0, 5),
#         "Omega": (0, 1),
#         "phi": (-torch.pi, torch.pi),
#         "tau": (0, 1.0)
#     },
#     "max_pulses": 10,
# }

# model = CompositePulseTransformerEncoder(**model_param)
# model.train()

# # Step 2: Create dummy input (batch of unitaries)
# B = 4
# I = torch.eye(2, dtype=torch.cfloat).unsqueeze(0).repeat(B, 1, 1)  # Identity unitaries
# U_target = I.clone().requires_grad_(False)

# # Step 3: Dummy optimizer
# trainer_params = {
#     "model" : model, "unitary_generator" : batched_unitary_generator,
#     "error_sampler": get_ore_error_distribution,
#     "fidelity_fn": fidelity,
#     "loss_fn": negative_log_loss,
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }

# trainer = CompositePulseTrainer(**trainer_params)

# # Step 4: Forward + Loss + Backward
# error_params_list = [{"delta_std" : delta_std} for delta_std in (0.1,)]

# trainer.train(U_targets, error_params_list=error_params_list, epochs=1, save_path="test/test2")



# # Step 5: Check all parameters have .grad
# print("Checking gradients:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         assert param.grad is not None, f"{name} has no gradient!"
#         print(f"✅ {name}: gradient exists with shape {param.grad.shape}")

# # Step 6: Save old weights and step optimizer
# old_params = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

# error_params_list = [{"delta_std" : delta_std} for delta_std in (0.1,)]

# trainer.train(U_targets, error_params_list=error_params_list, epochs=1, save_path="test/test2")





# # Step 7: Check if weights changed
# print("\nChecking weight updates:")
# for name, param in model.named_parameters():
#     if name in old_params:
#         changed = not torch.allclose(param, old_params[name])
#         print(f"{'✅' if changed else '❌'} {name}: {'updated' if changed else 'unchanged'}")




def Rx(theta):
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    return torch.matrix_exp(-1j * X * theta / 2)

def Ry(theta):
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    return torch.matrix_exp(-1j * Y * theta / 2)

def get_sigma_n(n):
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return n[0] * X + n[1] * Y + n[2] * Z


def test_euler_yxy_from_rotation_vector(batch_size=10000, tol=5e-3):
    # Random angles
    theta = torch.rand(batch_size) * math.pi
    phi = torch.rand(batch_size) * 0
    alpha = torch.rand(batch_size) * 2 * math.pi

    # Rotation axis (spherical coordinates)
    n_x = torch.sin(theta) * torch.cos(phi)
    n_y = torch.sin(theta) * torch.sin(phi)
    n_z = torch.cos(theta)
    n = torch.stack([n_x, n_y, n_z], dim=1)  # (B, 3)
    n = n / n.norm(dim=1, keepdim=True)

    # Rotation vector for the function: (n_x, n_y, n_z, alpha)
    rotation_vector = torch.cat([n, alpha.unsqueeze(1)], dim=1)  # (B, 4)

    # Input unitaries
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    sigma_n = n[:, 0, None, None] * X + n[:, 1, None, None] * Y + n[:, 2, None, None] * Z  # (B, 2, 2)
    alpha_half = alpha / 2
    U_input = torch.matrix_exp(-1j * sigma_n * alpha_half[:, None, None])  # (B, 2, 2)

    # Decompose
    euler_angles = UniversalQOCTransformer.euler_yxy_from_rotation_vector(rotation_vector)  # (B, 3)
    a, b, g = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    # Reconstruct unitaries
    def Rx(theta):
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=theta.device)
        return torch.matrix_exp(-1j * X * theta[:, None, None] / 2)

    def Ry(theta):
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=theta.device)
        return torch.matrix_exp(-1j * Y * theta[:, None, None] / 2)

    U_out = Ry(a) @ Rx(b) @ Ry(g)  # (B, 2, 2)

    print(U_input[0])
    print(U_out[0])

    # Check |Tr(U_out† U_input)|^2 == 4 (for 2x2 unitaries, d=2)
    d = 2
    inner = torch.einsum("bij,bij->b", U_out.conj().transpose(-2, -1), U_input)
    fidelity = (inner.abs() ** 2) / (d ** 2)

    print(f"Min fidelity: {fidelity.min().item():.8f}")
    print(f"Mean fidelity: {fidelity.mean().item():.8f}")
    print(f"Num failed: {torch.sum(fidelity < 1 - tol).item()} out of {batch_size}  ")
    num_failed = torch.sum(fidelity < 1 - tol).item()
    print(f"Num failed: {num_failed} out of {batch_size}")

    if num_failed > 0:
        failed_mask = fidelity < 1 - tol
        print("First few failing cases (theta, alpha):")
        print("theta:", theta[failed_mask][:10].cpu().numpy())
        print("alpha:", alpha[failed_mask][:10].cpu().numpy())
        print("U_input:", U_input[failed_mask][:1].cpu().numpy())
        print("U_out:", U_out[failed_mask][:1].cpu().numpy())

        f = U_input[failed_mask][0].cpu().numpy().conj().T @ U_out[failed_mask][0].cpu().numpy()
        print("First failing U_input† @ U_out:", np.abs(f[0,0] + f[1, 1]) **2 / (d ** 2))

    assert torch.allclose(fidelity, torch.ones_like(fidelity), atol=tol), "Some decompositions failed!"
    print("All tests passed.")


# Run the vectorized test
test_euler_yxy_from_rotation_vector(batch_size=10000)


def test_X_pi_pulse_robustness(num_samples=1000):
    import pandas as pd

    df = pd.read_csv("smoothing/X_pi_pulse.csv")
    # Columns: t (us), Omega_x (2pi MHz), Omega_y (2pi MHz), Omega_z (2pi MHz)

    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)

    t_vals = torch.tensor(df["t (us)"].values, dtype=torch.float64)
    Omega_x_vals = torch.tensor(df["Omega_x (2pi MHz)"].values, dtype=torch.float64)
    Omega_y_vals = torch.tensor(df["Omega_y (2pi MHz)"].values, dtype=torch.float64)

    sigma_x = sigma_x.to(torch.cdouble)
    sigma_y = sigma_y.to(torch.cdouble)
    sigma_z = sigma_z.to(torch.cdouble)

    # delta ~ 80 * N(0, 1), shape: (num_samples,)
    delta_samples = 2 * np.pi * 80.0 * torch.randn(num_samples, dtype=torch.float64)

    # Accumulate U = V_L @ ... @ V_1 for each sample simultaneously
    U = torch.eye(2, dtype=torch.cdouble).unsqueeze(0).expand(num_samples, 2, 2).clone()

    L = len(df)
    for j in range(L):
        Omega_x_j = 2 * np.pi * Omega_x_vals[j]  # rad/us
        Omega_y_j = 2 * np.pi * Omega_y_vals[j]  # rad/us
        t_j = t_vals[j]                            # us

        # H_j(delta) = 1/2 * (Omega_x_j * sigma_x + Omega_y_j * sigma_y + delta * sigma_z)
        H_common = 0.5 * (Omega_x_j * sigma_x + Omega_y_j * sigma_y)          # (2, 2)
        H_delta = 0.5 * delta_samples[:, None, None] * sigma_z                 # (num_samples, 2, 2)
        H_j = H_common.unsqueeze(0) + H_delta                                  # (num_samples, 2, 2)

        # V_j = exp(-i * H_j * t_j)
        V_j = torch.matrix_exp(-1j * H_j * t_j)                               # (num_samples, 2, 2)
        U = V_j @ U

    # Fidelity = (|Tr(U_out^dag @ sigma_x)|^2 + 2) / 6
    UdagSx = U.conj().transpose(-2, -1) @ sigma_x.unsqueeze(0)                # (num_samples, 2, 2)
    trace_vals = UdagSx[:, 0, 0] + UdagSx[:, 1, 1]                           # (num_samples,)
    fidelities = (trace_vals.abs() ** 2 + 2) / 6

    avg_fidelity = fidelities.mean().item()
    print(f"Average fidelity: {avg_fidelity:.4f} ({avg_fidelity * 100:.2f}%)")
    assert abs(avg_fidelity - 0.991) < 0.01, (
        f"Expected ~99.1% average fidelity, got {avg_fidelity * 100:.2f}%"
    )
    print("Test passed: X pi pulse average fidelity is ~99.1%")


test_X_pi_pulse_robustness()
