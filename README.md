# Robust Quantum Control with Composite Pulse Sequences

This project develops a machine learning framework for generating composite pulse sequences that implement a target quantum operation with high fidelity. It is specifically aimed to create pulses that are robust under strong static disorder (e.g., off-resonant errors). It leverages a transformer encoder model to output pulse sequences robust to errors sampled from a given distribution.

---

## 🧠 Objective and Problem Formulation

### Goal

Implement a target quantum unitary $U_{\text{target}}$ using a pulse sequence $[p_1, p_2, ..., p_L] \in \mathcal{P}^L$, where the resulting unitary $U_{\text{out}}$ is robust against a static error $\vec{\epsilon} \sim p_{\vec{\epsilon}}(\cdot |\vec{\Sigma})$. The primary objective is to optimize composite pulse sequence for a **large** disorder.

### Problem Input:

* Number of qubits $n$
* Target unitary $U_{\text{target}} \in \mathbb{C}^{2^n \times 2^n}$
* Pulse parameter space $\mathcal{P}$
* Static error model $\vec{\epsilon} \sim p_{\vec{\epsilon}}(\cdot |\vec{\Sigma})$ where $\vec{\Sigma}$ quantifies the standard deviation.
* Unitary generator $U_{\text{out}} \leftarrow g(p, \vec{\epsilon})$ that creates the unitary from pulse $p \in \mathcal{P}$ with error $\vec{\epsilon}$. 

### Problem Output:

* Length L pulse sequence $p_1, ... p_L$.

### Objective:

Maximize expected fidelity:

```math
\mathbb{E}_{\vec{\epsilon} \sim p(\cdot |\vec{\Sigma})}\left[ \frac{\left| \text{Tr}(U_{\text{out}}^{\dagger} U_{\text{target}}) \right|^2 + d}{d^2 + d}\right]
```

where 
```math
U_{\text{out}} = U_L \cdots U_1 \text{ and } U_i = \text{unitary\_generator}(p_i, \vec{\epsilon})
```

A transformer encoder model $f(U_{\text{target}}; \theta)$ is trained to generate the pulse sequence.


### Optimization Code:

The code utilizes an RL framework, where the environment is the error distribution, agent is the transformer model, and the action space is the pulse sequence. The following diagram is how the training code optimizes the fidelity (reward)

<p align="center">
  <img src="assets/training objective.png"  alt="training objective">
</p>

The key intuition for this project is to iteratively train the model from low to large disorder, which is known as curriculum learning in RL. The following is a pseudocode for model training

```{r, eval = FALSE}
train(unitary_generator, error_distribution, U_target):
    theta <- initial model parameter
    for error_param from small to large:
        - for each epoch
            - pulses <- f(U_target; theta) # model output
            - error <- error_distribution(error_param) # error
            - U_out <- unitary_generator(pulses, error)
            - loss_fn <- -log(E[fidelity(U_out, U_target)])
            - theta <- theta - eta * \partial_\theta loss_fn
```

To generate smoother pulse sequence, we execute the following post-processing and finetuning:

<p align="center">
  <img src="assets/training pipeline.png"  alt="training pipeline">
</p>

---

## 📁 Codebase Structure

This repository is organized into the following key directories:

### `model/`

Contains the core machine learning logic for composite pulse sequence generation:

* `model_encoder.py`: Defines the Transformer-based model architecture for generating pulse sequences.
* `trainer.py`: Implements the training loop and optimization logic for model learning.

### `weights/`

Stores pretrained model weights and the optimized pulse sequences:

* Use these files for direct inference without retraining.
* Includes sample pulse outputs in csv format.

### `train/`

Contains training scripts tailored to specific quantum systems:

* `single_qubit_phase_control_only/`: Scripts for training on single-qubit target unitaries. Pulse Sequence is series of resonant pulses.
* `two_qubit/`: Scripts for training on two-qubit operations (e.g., entangling gates).

You can configure the model and training settings via the `model_params.json` file in each subfolder.

### `visualize/`

Includes visualization utilities:

* Use these scripts to plot and analyze the learned composite pulse sequences.
* Helpful for inspecting fidelity contours, pulse robustness, and system behavior under error.

---

To get started, run a training script under `train/`, or load a pretrained model from `weights/` and use the visualization tools to analyze performance.


## 🚀 Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python train/single_qubit_phase_only/single_qubit_phase_control.py --num_epoch [num_epoch] --save_path [save_path]
```

* For pre-training, set "finetuning" to false. Otherwise, pretrained single-qubit control pulse is saved as "combined.pt"

---

## 📌 Notes

* Supports general $n$-qubit systems
* Pulse space $\mathcal{P}$ can be continuous (e.g., $\phi, t$)
* Custom loss is used such that it has zero gradient at $F = 1$ and sharp gradient at $F < 0.99$. The loss function is 

$$L(F; \tau=0.99, k=100) = \log(1 + \exp(-k \cdot (F - \tau))\cdot (1 - F)$$

---

## 📄 License

MIT License

## ✏️ Citation

If you use this work in academic research or teaching, please cite appropriately.
