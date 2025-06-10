import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from model import CompositePulseTransformerDecoder


class CompositePulseTrainer:
    """Trainer for :class:`CompositePulseTransformerDecoder`."""

    def __init__(
        self,
        model: CompositePulseTransformerDecoder,
        unitary_generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        error_sampler: Callable[[int], torch.Tensor],
        *,
        fidelity_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor, Callable, int], torch.Tensor] | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monte_carlo: int = 1000,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.unitary_generator = unitary_generator
        self.error_sampler = error_sampler
        self.fidelity_fn = fidelity_fn
        self.loss_fn = loss_fn 
        self.monte_carlo = monte_carlo
        self.device = device

        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # State tracking
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_pulses: torch.Tensor | None = None
        self.best_fidelity: float = 0.0

    # ------------------------------------------------------------------
    # Training loop utilities
    # ------------------------------------------------------------------

    def train_epoch(self, U_target: torch.Tensor, error_distribution) -> float:
        """One optimisation step (epoch).

        The Monte‑Carlo dimension is fused into the batch so the network is
        executed **once** per epoch, eliminating thousands of CUDA launches
        that previously dominated runtime.
        """
        self.model.train()
        # Back‑prop
        self.optimizer.zero_grad()

        U_target = U_target.to(self.device)

        # Forward pass once to obtain pulse parameters
        pulses = self.model(U_target)  # (B, L, P)

        # ──────────────────────────────────────────────────────────────
        # Vectorised Monte‑Carlo sampling
        # ──────────────────────────────────────────────────────────────
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)   # (Bm, L, P)
        targets_mc = U_target.repeat_interleave(self.monte_carlo, dim=0)  # (Bm, d, d)
        error = error_distribution(self.monte_carlo * U_target.shape[0]).to(self.device)                   # (Bm, …)

        U_out = self.unitary_generator(pulses_mc, error)              # (Bm, d, d)

        # print(U_out.shape, targets_mc.shape)


        loss = self.loss_fn(U_out, targets_mc, self.fidelity_fn, self.model.num_qubits)

        # print(self.fidelity_fn(U_out, targets_mc, self.model.num_qubits))

        
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().item())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, U_target_batch: torch.Tensor, error_distribution) -> float:
        """Compute mean fidelity on *U_target_batch* (no grad)."""
        self.model.eval()

        U_target = U_target_batch.to(self.device)
        pulses = self.model(U_target)
        error = error_distribution(U_target.shape[0]).to(self.device)
        U_out = self.unitary_generator(pulses, error)
        mean_fid = self.fidelity_fn(U_out, U_target, self.model.num_qubits).mean().item()
        return mean_fid

    # ------------------------------------------------------------------
    # Error helper
    # ------------------------------------------------------------------

    def get_error_distribution(self, *, error_params: Dict) -> Callable[[int], torch.Tensor]:
        """Return λ(batch_size) that samples *error* with the supplied parameters."""
        def _sampler(batch_size: int):
            return self.error_sampler(batch_size, **error_params)
        return _sampler

    # ------------------------------------------------------------------
    # Top‑level training orchestrator
    # ------------------------------------------------------------------

    def train(
        self,
        train_set: torch.Tensor,
        *,
        error_params_list: List[Dict],  # iterate from small → large error
        epochs: int = 100,
        save_path: str | Path | None = None,
    ) -> None:
        """Optimise *model*; keep weights with **highest fidelity**."""
        self.device = str(self.device)
        self.model.to(self.device)

        for error_params in error_params_list:
            self.best_fidelity = 0.0
            error_distribution = self.get_error_distribution(error_params=error_params)

            with tqdm(total=epochs, desc=f"ϵ = {error_params}", dynamic_ncols=True) as pbar:
                for epoch in range(1, epochs + 1):
                    train_loss = self.train_epoch(train_set, error_distribution)
                    eval_fid = self.evaluate(train_set, error_distribution)

                    # Track best model
                    if eval_fid > self.best_fidelity:
                        self.best_fidelity = eval_fid
                        self.best_state = {
                            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                        }
                        self.best_pulses = self.model(train_set.to(self.device)).detach().cpu()

                    pbar.set_postfix({
                        "epoch": epoch,
                        "loss": train_loss,
                        "fid": eval_fid,
                        "best": self.best_fidelity,
                    })
                    pbar.update(1)

            # Reload best weights after finishing current error‑band
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)

            # Persist
            if save_path is not None:
                tag = f"{save_path}_err_{str(error_params).replace(' ', '')}"
                self._save_weight(f"{tag}.pt")
                self._save_pulses(f"{tag}_pulses.pt", train_set)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_weight(self, path: str | Path) -> None:
        if self.best_state is None:
            raise RuntimeError("No trained weights recorded – call .train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_state, str(path))
        print(f"Weights saved → {path}")

    def _save_pulses(self, path: str | Path, unitaries: Sequence[torch.Tensor]) -> None:
        self.model.eval()
        with torch.no_grad():
            pulses = self.model(unitaries.to(self.device)).cpu()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pulses, str(path))
        print(f"Pulse parameters saved → {path}")
