"""Curriculum Monte-Carlo trainer.

One trainer handles any `BaseController`. Monte-Carlo error samples are fused
into the batch dim, so the model runs once per optimisation step.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .fidelity import fidelity as default_fidelity
from .models.base import BaseController


UnitaryGenerator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ErrorSampler = Callable[[int], torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


class Trainer:
    def __init__(
        self,
        model: BaseController,
        unitary_generator: UnitaryGenerator,
        loss_fn: LossFn,
        *,
        monte_carlo: int = 1000,
        learning_rate: float = 3e-5,
        grad_clip: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.unitary_generator = unitary_generator
        self.loss_fn = loss_fn
        self.monte_carlo = monte_carlo
        self.grad_clip = grad_clip
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[trainer] model '{type(self.model).__name__}' has {n_params:,} parameters")

        self.best_state: Dict[str, torch.Tensor] | None = None
        self.best_fidelity: float = 0.0

    # --------------------------------------------------------------- step
    def _rollout(self, rot_vec: torch.Tensor, U_target: torch.Tensor,
                 error_sampler: ErrorSampler) -> torch.Tensor:
        """Forward pulses and propagator once over MC-expanded batch. Returns (U_out, U_target_mc)."""
        pulses = self.model(rot_vec)
        B = rot_vec.shape[0]
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)
        target_mc = U_target.repeat_interleave(self.monte_carlo, dim=0)
        error = error_sampler(self.monte_carlo * B).to(self.device)
        U_out = self.unitary_generator(pulses_mc, error)
        return U_out, target_mc

    def train_step(self, rot_vec: torch.Tensor, U_target: torch.Tensor,
                   error_sampler: ErrorSampler) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        U_out, target_mc = self._rollout(rot_vec, U_target, error_sampler)
        loss = self.loss_fn(U_out, target_mc, self.model.num_qubits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self, rot_vec: torch.Tensor, U_target: torch.Tensor,
                 error_sampler: ErrorSampler) -> float:
        self.model.eval()
        U_out, target_mc = self._rollout(rot_vec, U_target, error_sampler)
        return default_fidelity(U_out, target_mc, self.model.num_qubits).mean().item()

    # --------------------------------------------------------------- loop
    def train(
        self,
        train_rot_vec: torch.Tensor,
        train_U: torch.Tensor,
        eval_rot_vec: torch.Tensor,
        eval_U: torch.Tensor,
        curriculum: List[Dict[str, float]],
        error_sampler_factory: Callable[[Dict], ErrorSampler],
        epochs: int,
        batch_size: int,
        save_dir: str | Path | None = None,
    ) -> None:
        save_dir_path = Path(save_dir) if save_dir else None
        train_rot_batches = _chunk(train_rot_vec, batch_size)
        train_U_batches = _chunk(train_U, batch_size)
        eval_rot_batches = _chunk(eval_rot_vec, batch_size)
        eval_U_batches = _chunk(eval_U, batch_size)

        for params in curriculum:
            self.best_fidelity = 0.0
            self.best_state = None
            sampler = error_sampler_factory(params)

            with tqdm(total=epochs, desc=f"ε={params}", dynamic_ncols=True) as pbar:
                for epoch in range(1, epochs + 1):
                    losses, fids = [], []
                    for r, U in zip(train_rot_batches, train_U_batches):
                        losses.append(self.train_step(r.to(self.device), U.to(self.device), sampler))
                    for r, U in zip(eval_rot_batches, eval_U_batches):
                        fids.append(self.evaluate(r.to(self.device), U.to(self.device), sampler))

                    loss_mean = float(np.mean(losses))
                    fid_mean = float(np.mean(fids))
                    if fid_mean > self.best_fidelity:
                        self.best_fidelity = fid_mean
                        self.best_state = _snapshot_state(self.model)

                    pbar.set_postfix(loss=loss_mean, fid=fid_mean, best=self.best_fidelity)
                    pbar.update(1)

            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)

            if save_dir_path is not None:
                tag = _param_tag(params)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_state, save_dir_path / f"{tag}.pt")
                with torch.no_grad():
                    pulses = self.model(train_rot_vec.to(self.device)).cpu()
                torch.save(pulses, save_dir_path / f"{tag}_pulses.pt")
                print(f"[trainer] saved weights → {save_dir_path / f'{tag}.pt'}")


def _snapshot_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in copy.deepcopy(model.state_dict()).items()}


def _chunk(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    N = x.shape[0]
    if N % batch_size != 0:
        raise ValueError(f"dataset size {N} not divisible by batch_size {batch_size}")
    return x.view(N // batch_size, batch_size, *x.shape[1:])


def _param_tag(params: Dict[str, float]) -> str:
    return "err_" + "_".join(f"{k}{v:.3f}" for k, v in params.items())
