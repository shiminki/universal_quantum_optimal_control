"""YAML-driven run configuration.

Example:

    model:
      type: transformer
      num_qubits: 1
      pulse_space: {phi: [-3.15, 3.15], tau: [0.1, 0.5]}
      max_pulses: 400
      d_model: 512
      n_layers: 8
      n_heads: 16
      dropout: 0.1

    training:
      batch_size: 8
      epochs: 50
      monte_carlo: 1000
      learning_rate: 3.0e-5
      loss: sharp_rotation
      loss_params: {rotation_weight: 1.0}
      error_sampler: ore_mhz
      curriculum:
        - {delta_std_mhz: 30.0}
        - {delta_std_mhz: 60.0}
        - {delta_std_mhz: 90.0}

    physics:
      omega_mhz: 30.0

    bandwidth:
      mode: filter          # none | filter | penalty | both
      cutoff_mhz: 300.0
      penalty_weight: 1.0

    dataset:
      train_size: 10000
      eval_size: 1000
      random: true

    seed: 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    monte_carlo: int = 1000
    learning_rate: float = 3e-5
    grad_clip: float = 1.0
    loss: str = "sharp"
    loss_params: Dict[str, float] = field(default_factory=dict)
    error_sampler: str = "ore_ple"
    curriculum: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class PhysicsConfig:
    """Physical hardware parameters mapping dimensionless units to MHz/µs."""
    omega_mhz: float = 30.0


@dataclass
class BandwidthConfig:
    """AWG bandwidth constraint (see uqoc.bandwidth)."""
    mode: str = "none"          # none | filter | penalty | both
    cutoff_mhz: float = 300.0
    penalty_weight: float = 1.0
    max_radius: int = 32
    oversample: int = 8         # fine-grid points per segment for filter mode


@dataclass
class DatasetConfig:
    train_size: int = 10000
    eval_size: int = 1000
    random: bool = True


@dataclass
class Config:
    model: Dict[str, Any]
    training: TrainingConfig
    dataset: DatasetConfig
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    seed: int = 0


def _coerce_pulse_space(model: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pulse_space ranges to tuples so downstream code can rely on that."""
    if "pulse_space" in model:
        model["pulse_space"] = {k: tuple(v) for k, v in model["pulse_space"].items()}
    return model


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    model = _coerce_pulse_space(raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    dataset = DatasetConfig(**raw.get("dataset", {}))
    physics = PhysicsConfig(**raw.get("physics", {}))
    bandwidth = BandwidthConfig(**raw.get("bandwidth", {}))
    return Config(model=model, training=training, dataset=dataset,
                  physics=physics, bandwidth=bandwidth, seed=raw.get("seed", 0))
