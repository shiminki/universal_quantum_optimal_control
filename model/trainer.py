import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List, Union
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import matplotlib.pyplot as plt

from model.model import CompositePulseTransformerEncoder


__all__ = ["CompositePulseTrainer"]


class CompositePulseTrainer:
    """Trainer for :class:`CompositePulseTransformerDecoder`."""

    def __init__(
        self,
        # model: Union[CompositePulseTransformerDecoder, CompositePulseTransformerEncoder],
        model: CompositePulseTransformerEncoder,
        unitary_generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        error_sampler: Callable[[int], torch.Tensor],
        *,
        fidelity_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor, Callable, int], torch.Tensor] | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monte_carlo: int = 1000,
        device: str = "cuda",
    ) -> None:
        print(f"Total parameter: {sum(p.numel() for p in model.parameters())}")
        self.model = model.to(device)
        self.unitary_generator = unitary_generator
        self.error_sampler = error_sampler
        self.fidelity_fn = fidelity_fn
        self.loss_fn = loss_fn 
        self.monte_carlo = monte_carlo
        self.device = device

        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=3e-5)

        # State tracking
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_pulses: torch.Tensor | None = None
        self.best_fidelity: float = 0.0


    # ------------------------------------------------------------------
    # Training loop utilities
    # ------------------------------------------------------------------

    def train_epoch(self, U_emb_batch: torch.Tensor, U_target_batch: torch.Tensor, error_distribution) -> float:
        """One optimisation step (epoch).

        U_emb is the SCORE expansion of U_target

        The Monte‑Carlo dimension is fused into the batch so the network is
        executed **once** per epoch, eliminating thousands of CUDA launches
        that previously dominated runtime.
        """
        self.model.train()
        # Back‑prop
        self.optimizer.zero_grad()

        U_emb = U_emb_batch.to(self.device)
        U_target = U_target_batch.to(self.device)

        # Forward pass once to obtain pulse parameters
        pulses = self.model(U_emb)  # (B, L, P)

        # ──────────────────────────────────────────────────────────────
        # Vectorised Monte‑Carlo sampling
        # ──────────────────────────────────────────────────────────────
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)   # (Bm, L, P)
        targets_mc = U_target.repeat_interleave(self.monte_carlo, dim=0)  # (Bm, d, d)
        error = error_distribution(self.monte_carlo * U_target.shape[0]).to(self.device)                   # (Bm, …)

        U_out = self.unitary_generator(pulses_mc, error)              # (Bm, d, d)

        # print(U_out.shape, targets_mc.shape)

        loss = self.loss_fn(U_out, targets_mc, self.fidelity_fn, self.model.num_qubits)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.detach().item())
    
   
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, U_emb_batch: torch.Tensor, U_target_batch: torch.Tensor, error_distribution) -> float:
        """Compute mean fidelity on *U_target_batch* (no grad)."""
        self.model.eval()

        U_emb = U_emb_batch.to(self.device)
        U_target = U_target_batch.to(self.device)
        pulses = self.model(U_emb)
        
        # ──────────────────────────────────────────────────────────────
        # Vectorised Monte‑Carlo sampling
        # ──────────────────────────────────────────────────────────────
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)   # (Bm, L, P)
        targets_mc = U_target.repeat_interleave(self.monte_carlo, dim=0)  # (Bm, d, d)
        error = error_distribution(self.monte_carlo * U_target.shape[0]).to(self.device)                   # (Bm, …)

        U_out = self.unitary_generator(pulses_mc, error)              # (Bm, d, d)


        mean_fid = self.fidelity_fn(U_out, targets_mc, self.model.num_qubits).mean().item()
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
        eval_set: torch.Tensor,
        error_params_list: List[Dict],  # iterate from small → large error
        eval_error_param: Dict = None,
        epochs: int = 100,
        save_path: str | Path | None = None,
        plot: bool = False
    ) -> None:
        """Optimise *model*; keep weights with **highest fidelity**."""
        self.device = str(self.device)
        self.model.to(self.device)

        #########################
        # Universal gate version
        #########################

        L = train_set.shape[0]

        train_set_batch = train_set.view(10, L//10, 3, 2, 2)
        eval_set_batch = eval_set.view(10, L//10, 2, 2)

        #########################

        for error_params in error_params_list:
            self.best_fidelity = 0.0
            error_distribution = self.get_error_distribution(error_params=error_params)
            if eval_error_param is not None:
                eval_dist = self.get_error_distribution(error_params = eval_error_param)
            else:
                eval_dist = error_distribution

            fidelity_list = []

            with tqdm(total=epochs, desc=f"ϵ = {error_params}", dynamic_ncols=True) as pbar:
                for epoch in range(1, epochs + 1):
                    train_loss_list = []
                    eval_fid_list = []

                    for train_set, eval_set in tqdm(list(zip(train_set_batch, eval_set_batch))):
                        train_loss = self.train_epoch(train_set, eval_set, error_distribution) 
                        eval_fid = self.evaluate(train_set, eval_set, eval_dist)
                        train_loss_list.append(train_loss)
                        eval_fid_list.append(eval_fid)

                    train_loss = torch.mean(train_loss_list).item()
                    eval_fid = torch.mean(eval_fid_list).item()

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

                    fidelity_list.append(eval_fid)
                
                if plot:
                    plt.figure(figsize=(8, 4))
                    plt.plot(range(1, epochs + 1), fidelity_list, marker='o')
                    plt.xlabel("Epoch")
                    plt.ylabel("Evaluation Fidelity")
                    plt.title(f"Evaluation Fidelity vs Epoch with \nError: {error_params}")
                    plt.grid(True)
                    plt.tight_layout()
                    tag = os.path.join(save_path, f"err_{str(error_params).replace(' ', '')}")
                    fig_path = f"{tag}_loss_plot.png"
                    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(fig_path)
                    # plt.show()


            # Reload best weights after finishing current error‑band
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)

            # Persist
            if save_path is not None:
                tag = os.path.join(save_path, f"err_{str(error_params).replace(' ', '')}")
                self._save_weight(f"{tag}.pt")
                self._save_pulses(f"{tag}_pulses.pt", train_set)

    @torch.no_grad()
    def get_average_fidelity(
        self,
        train_set: torch.Tensor,
        error_params: Dict
    ):
        self.model.eval()
        self.device = str(self.device)
        self.model.to(self.device)
        error_distribution = self.get_error_distribution(error_params=error_params)
        eval_fid = self.evaluate(train_set, error_distribution)

        return eval_fid

        

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
