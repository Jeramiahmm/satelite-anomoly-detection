"""
Federated Aggregator — Secure FedAvg over mTLS.

Implements the Federated Averaging algorithm for the Astraea-1 constellation.
Only "Weight Deltas" (Δw) are exchanged — raw telemetry never leaves any node.

The aggregator runs on a designated node (or rotates via leader election).
It collects weight deltas from all trusted (non-revoked) peers, computes
the averaged global update, and broadcasts the new global weights.

Security invariants:
    1. Deltas are transmitted over mTLS-secured ISL channels.
    2. Nodes on the CRL are excluded from aggregation.
    3. Weight deltas are clipped to prevent gradient poisoning.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
           from Decentralized Data" (2017) — adapted for constrained ISL links.
"""

from __future__ import annotations

import copy
import json
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from ..ml.vae import TelemetryVAE

logger = logging.getLogger(__name__)

# Gradient clipping bound to prevent poisoning attacks
DELTA_CLIP_NORM = 10.0


@dataclass
class FederationRound:
    """Metadata for a single federation round."""

    round_id: int
    started_at: float
    completed_at: float = 0.0
    participants: list = field(default_factory=list)
    excluded_nodes: list = field(default_factory=list)
    avg_delta_norm: float = 0.0


class FederatedAggregator:
    """
    Secure FedAvg coordinator for the Astraea-1 constellation.

    Aggregation policy:
        w_global_new = w_global + (1/n) * Σ clip(Δw_i, DELTA_CLIP_NORM)

    Where n = number of trusted participants in this round.
    """

    def __init__(
        self,
        global_model: TelemetryVAE,
        min_participants: int = 2,
        clip_norm: float = DELTA_CLIP_NORM,
    ):
        """
        Args:
            global_model: The current global VAE model.
            min_participants: Minimum nodes required to run a round.
            clip_norm: L2 norm bound for delta clipping.
        """
        self.global_model = global_model
        self.global_state = copy.deepcopy(global_model.state_dict())
        self.min_participants = min_participants
        self.clip_norm = clip_norm
        self.round_id = 0
        self.history: List[FederationRound] = []

        self._pending_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        self._lock = threading.Lock()

    def submit_delta(
        self,
        node_id: str,
        delta: Dict[str, torch.Tensor],
        revoked_serials: frozenset = frozenset(),
        node_serial: Optional[int] = None,
    ) -> bool:
        """
        Submit a weight delta from a satellite node.

        Args:
            node_id: The submitting node's identifier.
            delta: Dict mapping parameter names to delta tensors.
            revoked_serials: Current CRL — used to reject revoked nodes.
            node_serial: The node's certificate serial number.

        Returns:
            True if accepted, False if rejected (e.g., node is revoked).
        """
        if node_serial is not None and node_serial in revoked_serials:
            logger.warning(
                "REJECTED delta from revoked node %s (serial=%d)",
                node_id,
                node_serial,
            )
            return False

        # Clip the delta to prevent gradient poisoning
        clipped_delta = self._clip_delta(delta)

        with self._lock:
            self._pending_deltas[node_id] = clipped_delta
            logger.info(
                "Accepted delta from %s  (%d/%d pending)",
                node_id,
                len(self._pending_deltas),
                self.min_participants,
            )
        return True

    def aggregate(
        self, excluded_nodes: Optional[Set[str]] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Run a FedAvg aggregation round.

        Returns the new global state dict, or None if insufficient participants.
        """
        excluded = excluded_nodes or set()

        with self._lock:
            # Filter out excluded/revoked nodes
            eligible = {
                nid: delta
                for nid, delta in self._pending_deltas.items()
                if nid not in excluded
            }

            if len(eligible) < self.min_participants:
                logger.warning(
                    "Insufficient participants: %d/%d (need %d)",
                    len(eligible),
                    len(self._pending_deltas),
                    self.min_participants,
                )
                return None

            self.round_id += 1
            round_meta = FederationRound(
                round_id=self.round_id,
                started_at=time.time(),
                participants=list(eligible.keys()),
                excluded_nodes=list(excluded),
            )

            # FedAvg: new_global = global + (1/n) * Σ Δw_i
            n = len(eligible)
            avg_delta: Dict[str, torch.Tensor] = {}
            total_norm = 0.0

            for key in self.global_state:
                stacked = torch.stack([d[key].float() for d in eligible.values()])
                avg_delta[key] = stacked.mean(dim=0)
                total_norm += avg_delta[key].norm().item()

            # Apply aggregated delta to global state
            new_global = {}
            for key in self.global_state:
                new_global[key] = self.global_state[key] + avg_delta[key]

            self.global_state = new_global
            self.global_model.load_state_dict(new_global)

            # Record round metadata
            round_meta.completed_at = time.time()
            round_meta.avg_delta_norm = total_norm / len(avg_delta)
            self.history.append(round_meta)

            # Clear pending deltas for next round
            self._pending_deltas.clear()

        logger.info(
            "Federation round %d complete: %d participants  "
            "avg_delta_norm=%.4f  duration=%.3fs",
            round_meta.round_id,
            n,
            round_meta.avg_delta_norm,
            round_meta.completed_at - round_meta.started_at,
        )

        return copy.deepcopy(new_global)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of the current global model state."""
        with self._lock:
            return copy.deepcopy(self.global_state)

    def _clip_delta(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip weight delta by L2 norm to bound gradient poisoning."""
        total_norm = 0.0
        for v in delta.values():
            total_norm += v.float().norm().item() ** 2
        total_norm = total_norm**0.5

        if total_norm > self.clip_norm:
            scale = self.clip_norm / total_norm
            logger.warning(
                "Clipping delta: norm=%.4f > clip=%.4f  scale=%.4f",
                total_norm,
                self.clip_norm,
                scale,
            )
            return {k: v * scale for k, v in delta.items()}
        return delta

    def serialize_global_state(self) -> bytes:
        """Serialize global state for ISL broadcast."""
        buf = io.BytesIO()
        torch.save(self.get_global_state(), buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_global_state(data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize global state received from ISL."""
        buf = io.BytesIO(data)
        return torch.load(buf, weights_only=True)
