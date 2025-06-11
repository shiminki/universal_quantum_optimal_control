from model_decoder import CompositePulseTransformerDecoder
from trainer import CompositePulseTrainer
from single_qubit_script import *
import torch

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)


model_param = {
    "num_qubits": 1,
    "pulse_space": {
        "Delta": (0, 5), "Omega": (0, 1), "phi": (-torch.pi, torch.pi), "tau": (0, 1.0)
    },
    "max_pulses": 10,
}

model = CompositePulseTransformerDecoder(**model_param)


U_targets = torch.stack([
    _I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU
])


pulses = model(U_targets)

print(pulses)
print(pulses.shape)




###########################
# Test ####################
###########################

# Step 1: Initialize model
model_param = {
    "num_qubits": 1,
    "pulse_space": {
        "Delta": (0, 5),
        "Omega": (0, 1),
        "phi": (-torch.pi, torch.pi),
        "tau": (0, 1.0)
    },
    "max_pulses": 10,
}

model = CompositePulseTransformerDecoder(**model_param)
model.train()

# Step 2: Create dummy input (batch of unitaries)
B = 4
I = torch.eye(2, dtype=torch.cfloat).unsqueeze(0).repeat(B, 1, 1)  # Identity unitaries
U_target = I.clone().requires_grad_(False)

# Step 3: Dummy optimizer
trainer_params = {
    "model" : model, "unitary_generator" : batched_unitary_generator,
    "error_sampler": get_ore_error_distribution,
    "fidelity_fn": fidelity,
    "loss_fn": negative_log_loss,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

trainer = CompositePulseTrainer(**trainer_params)

# Step 4: Forward + Loss + Backward
error_params_list = [{"delta_std" : delta_std} for delta_std in (0.1,)]

trainer.train(U_targets, error_params_list=error_params_list, epochs=1, save_path="test/test2")



# Step 5: Check all parameters have .grad
print("Checking gradients:")
for name, param in model.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"{name} has no gradient!"
        print(f"✅ {name}: gradient exists with shape {param.grad.shape}")

# Step 6: Save old weights and step optimizer
old_params = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

error_params_list = [{"delta_std" : delta_std} for delta_std in (0.1,)]

trainer.train(U_targets, error_params_list=error_params_list, epochs=1, save_path="test/test2")





# Step 7: Check if weights changed
print("\nChecking weight updates:")
for name, param in model.named_parameters():
    if name in old_params:
        changed = not torch.allclose(param, old_params[name])
        print(f"{'✅' if changed else '❌'} {name}: {'updated' if changed else 'unchanged'}")

