"""
Federated Aggregator — Standalone Entry Point
===============================================

This is the top-level aggregator script (Deliverable #4).

It can run either:
  1. As a standalone process coordinating federation rounds
  2. Embedded in a satellite node (when ASTRAEA_IS_AGGREGATOR=true)

The aggregator collects weight deltas from all trusted nodes over
mTLS-secured ISL channels, performs FedAvg with gradient clipping,
and broadcasts the updated global model.

Security invariants:
  - Nodes on the CRL are excluded from aggregation
  - Weight deltas are L2-norm clipped to prevent gradient poisoning
  - All communication is mTLS-wrapped (never plaintext)

Usage:
    python federated_aggregator.py
    python federated_aggregator.py --min-participants 3 --clip-norm 5.0
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import logging
import signal
import sys
import time
from typing import Dict, Set

import torch

from astraea.crypto.identity import (
    CertificateAuthority,
    generate_ca,
    issue_node_certificate,
    persist_identity,
)
from astraea.crypto.crl_manager import CRLCache
from astraea.federation.aggregator import FederatedAggregator
from astraea.messaging.isl import (
    ISLBus,
    TOPIC_CRL_SYNC,
    TOPIC_FEDERATION_DELTAS,
    TOPIC_FEDERATION_GLOBAL,
)
from astraea.ml.vae import TelemetryVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("aggregator")


class AggregatorService:
    """
    Standalone federation aggregator service.

    Listens for weight delta submissions, validates them against the CRL,
    runs FedAvg when enough participants have contributed, and broadcasts
    the updated global model to the constellation.
    """

    def __init__(
        self,
        node_id: str = "aggregator-01",
        nats_url: str = "nats://nats-server:4222",
        min_participants: int = 2,
        clip_norm: float = 10.0,
        round_timeout_s: float = 60.0,
    ):
        self.node_id = node_id
        self.nats_url = nats_url
        self.round_timeout = round_timeout_s

        # Crypto
        self.ca = generate_ca()
        self.identity = issue_node_certificate(self.ca, node_id)
        self.crl = CRLCache()

        # Global model
        self.global_model = TelemetryVAE()
        self.aggregator = FederatedAggregator(
            self.global_model,
            min_participants=min_participants,
            clip_norm=clip_norm,
        )

        # Messaging
        self.isl = ISLBus(node_id=node_id, nats_url=nats_url)

        # State
        self._running = False
        self._revoked_nodes: Set[str] = set()
        self._round_count = 0

    async def start(self) -> None:
        """Start the aggregator service."""
        logger.info("=" * 60)
        logger.info("  ASTRAEA-1 FEDERATION AGGREGATOR: %s", self.node_id)
        logger.info("=" * 60)
        logger.info("  NATS URL:          %s", self.nats_url)
        logger.info(
            "  Min Participants:  %d", self.aggregator.min_participants
        )
        logger.info("  Clip Norm:         %.1f", self.aggregator.clip_norm)
        logger.info("  Round Timeout:     %.1fs", self.round_timeout)
        logger.info("=" * 60)

        await self.isl.connect()

        # Subscribe to delta submissions and CRL updates
        await self.isl.subscribe(
            TOPIC_FEDERATION_DELTAS, self._on_delta_received
        )
        await self.isl.subscribe(TOPIC_CRL_SYNC, self._on_crl_sync)

        self._running = True
        logger.info("Aggregator ONLINE — waiting for delta submissions...")

        # Run periodic aggregation check
        while self._running:
            await asyncio.sleep(self.round_timeout)
            await self._try_aggregate()

    async def _on_delta_received(self, msg) -> None:
        """Handle incoming weight delta from a satellite node."""
        try:
            buf = io.BytesIO(msg.data)
            payload = torch.load(buf, weights_only=False)
            node_id = payload["node_id"]
            delta = payload["delta"]

            # Check CRL before accepting
            accepted = self.aggregator.submit_delta(
                node_id,
                delta,
                revoked_serials=self.crl.get_all_revoked(),
            )

            if accepted:
                logger.info(
                    "Delta accepted from %s  (pending: %d)",
                    node_id,
                    len(self.aggregator._pending_deltas),
                )
                # Try immediate aggregation if threshold met
                await self._try_aggregate()

        except Exception:
            logger.exception("Error processing delta submission")

    async def _on_crl_sync(self, msg) -> None:
        """Handle CRL synchronization from the constellation."""
        try:
            revoked = CRLCache.deserialize_broadcast(msg.data)
            added = self.crl.merge_from_broadcast(revoked, source="peer")
            if added > 0:
                logger.warning(
                    "CRL updated: %d new revocations (total: %d)",
                    added,
                    len(self.crl.get_all_revoked()),
                )
        except Exception:
            logger.exception("Error processing CRL sync")

    async def _try_aggregate(self) -> None:
        """Attempt a federation round if enough deltas are pending."""
        result = self.aggregator.aggregate(
            excluded_nodes=self._revoked_nodes
        )
        if result is not None:
            self._round_count += 1
            logger.info(
                "Federation round %d complete — broadcasting global model",
                self._round_count,
            )
            # Broadcast to constellation
            data = self.aggregator.serialize_global_state()
            await self.isl.publish(TOPIC_FEDERATION_GLOBAL, data)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self.isl.close()
        logger.info("Aggregator OFFLINE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astraea-1 Federated Aggregator"
    )
    parser.add_argument(
        "--node-id",
        default="aggregator-01",
        help="Aggregator node ID",
    )
    parser.add_argument(
        "--nats-url",
        default="nats://nats-server:4222",
        help="NATS server URL",
    )
    parser.add_argument(
        "--min-participants",
        type=int,
        default=2,
        help="Minimum participants for aggregation",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=10.0,
        help="L2 norm clip bound for weight deltas",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=60.0,
        help="Seconds between aggregation attempts",
    )
    args = parser.parse_args()

    service = AggregatorService(
        node_id=args.node_id,
        nats_url=args.nats_url,
        min_participants=args.min_participants,
        clip_norm=args.clip_norm,
        round_timeout_s=args.round_timeout,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        loop.create_task(service.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(service.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
