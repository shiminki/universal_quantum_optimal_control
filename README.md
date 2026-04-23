# Robust Quantum Control with Composite Pulse Sequences

Neural-network controllers that output composite `(phi, tau)` pulse sequences
implementing a target SU(2) gate, optimised for robustness to static disorder
(off-resonant detuning + pulse-length error).

Two controller families share the same training/evaluation stack:

* **`TransformerController`** — transformer encoder over SCORE-embedded Euler
  decomposition of the target, projected to a length-`max_pulses` sequence.
* **`DeepNNController`** — dense MLP directly mapping a reduced rotation vector
  to `num_pulses` pulse parameters.

## Problem

Target unitary `U_target ∈ SU(2)`. Static errors
`ε = (δ, ε) ~ p(·|Σ)` perturb the control Hamiltonian
`H(t) = 0.5·(1+ε)·(cos φ X + sin φ Y + δ Z)`.
A controller `f(U_target; θ)` emits a pulse sequence, and the propagator
`U_out = U_L ⋯ U_1` is compared to the target. Training maximises the
Haar-averaged entanglement fidelity across a curriculum of increasing disorder:

```
    F = E_ε[ (|Tr(U_out† U_target)|² + d) / (d(d+1)) ]
```

## Layout

```
src/uqoc/
  quantum.py          Paulis (cached), rotation_unitary, Y-X-Y Euler, SCORE, fidelity
  propagator.py       Closed-form (B, L, 2) → (B, 2, 2) log-depth composite
  errors.py           ORE / ORE+PLE samplers (registered by name)
  fidelity.py         {neg_log, infidelity, sharp} losses (registered by name)
  dataset.py          Random/grid SU(2) target sampler
  config.py           YAML → dataclass
  utils.py            set_seed, default_device
  models/
    base.py           BaseController + MODEL_REGISTRY
    transformer.py    TransformerController
    deep_nn.py        DeepNNController
  pipeline.py         Inference wrapper (rotation vec or unitary)
  trainer.py          Curriculum Monte-Carlo trainer (model-agnostic)

scripts/
  train.py            python scripts/train.py --config configs/x.yaml --save-dir outputs/x
  evaluate.py         python scripts/evaluate.py --config ... --checkpoint ...
  app.py              Gradio demo (loads HF weights by default)

configs/              YAML run configs
demo_universal/
  weight/*.pt         Pretrained checkpoints
  config/*.yaml       Matching architecture definitions

visualize/
  plots.py            Fidelity contour, pulse-param, fidelity-vs-std
  bloch_video.py      Bloch-sphere ensemble evolution animation

baselines/
  true_grape.py       Direct-parameter GRAPE (nn.Parameter, single target)

smoothing/            Low-pass filter for hardware-constrained pulses
tests/                pytest suite
```

## Install

```bash
pip install -e .              # core
pip install -e .[viz]         # + plotting (pwlf, qutip)
pip install -e .[app]         # + Gradio UI + HuggingFace
pip install -e .[dev]         # + pytest
```

## Train

```bash
python scripts/train.py --config configs/transformer_len100.yaml \
                        --save-dir outputs/transformer_len100
```

Configs pick the model, loss, error sampler, and curriculum. To add a new
controller, subclass `BaseController`, decorate with `@register("my_name")`,
and reference it in a YAML `model.type`.

## Evaluate / inspect

```bash
# Mean fidelity under the training curriculum
python scripts/evaluate.py --config configs/transformer_len100.yaml \
                           --checkpoint outputs/transformer_len100/err_delta_std1.000_epsilon_std0.050.pt

# Interactive Gradio demo (pulls weights from HuggingFace)
python scripts/app.py
```

## Tests

```bash
pytest tests/ -q
```

## License

MIT.
