"""
Local VAE Trainer.

Each satellite node trains a local VAE on its own telemetry stream.
Only weight deltas are shared with the federation — never raw telemetry.
This module handles the local training loop and weight delta computation.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import torch
import torch.optim as optim

from .telemetry import generate_telemetry_batch, normalize_telemetry, AttackMode
from .vae import TelemetryVAE, vae_loss

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Manages local training of the VAE and computes weight deltas for federation.
    """

    def __init__(
        self,
        model: TelemetryVAE,
        lr: float = 1e-3,
        kl_weight: float = 0.5,
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.kl_weight = kl_weight
        self._global_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.epoch_count = 0
        self.total_loss = 0.0

    def snapshot_global_weights(self) -> None:
        """Snapshot current model weights as the 'global' baseline."""
        self._global_snapshot = copy.deepcopy(self.model.state_dict())

    def train_epoch(
        self,
        batch_size: int = 64,
        num_batches: int = 10,
        time_step: int = 0,
    ) -> float:
        """
        Train for one local epoch on synthetic nominal telemetry.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0

        for _ in range(num_batches):
            raw = generate_telemetry_batch(
                batch_size, AttackMode.NOMINAL, time_step=time_step
            )
            x = normalize_telemetry(raw)

            self.optimizer.zero_grad()
            x_hat, mu, log_var = self.model(x)
            loss = vae_loss(x, x_hat, mu, log_var, self.kl_weight)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        self.epoch_count += 1
        self.total_loss = avg_loss
        logger.debug("Local epoch %d  avg_loss=%.6f", self.epoch_count, avg_loss)
        return avg_loss

    def compute_weight_delta(self) -> Dict[str, torch.Tensor]:
        """
        Compute Δw = w_local - w_global.

        This is the ONLY data shared with the federation.
        Raw telemetry NEVER leaves the node.
        """
        if self._global_snapshot is None:
            raise RuntimeError(
                "No global snapshot — call snapshot_global_weights() first"
            )
        current = self.model.state_dict()
        delta = {}
        for key in current:
            delta[key] = current[key] - self._global_snapshot[key]
        return delta

    def apply_global_weights(self, global_state: Dict[str, torch.Tensor]) -> None:
        """Apply aggregated global weights from the federation."""
        self.model.load_state_dict(global_state)
        self.snapshot_global_weights()
        logger.info("Applied global weights from federation round")
