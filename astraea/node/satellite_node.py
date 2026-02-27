"""
Satellite Node — Core Runtime for Astraea-1.

This is the primary executable for each satellite in the constellation.
It integrates all subsystems into a single asyncio event loop:

    1. VAE Inference Engine   — Real-time anomaly detection on telemetry
    2. mTLS Identity          — SPIFFE-inspired ephemeral certificates
    3. Policy Decision Point  — Zero-Trust evaluation and revocation
    4. Federated Trainer      — Local training + weight delta exchange
    5. Link-State Router      — Trust-weighted multi-hop routing
    6. ISL Message Bus        — NATS-backed inter-satellite communication

Lifecycle:
    boot() → identity_bootstrap() → [telemetry_loop | federation_loop |
              routing_loop | cert_rotation_loop] → shutdown()
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..crypto.identity import (
    CERT_TTL_SECONDS,
    CertificateAuthority,
    NodeIdentity,
    create_mtls_client_context,
    create_mtls_server_context,
    generate_ca,
    issue_node_certificate,
    persist_identity,
    extract_spiffe_id,
)
from ..crypto.crl_manager import CRLCache
from ..federation.aggregator import FederatedAggregator
from ..ml.anomaly_detector import AnomalyDetector
from ..ml.telemetry import (
    AttackMode,
    generate_telemetry_batch,
    normalize_telemetry,
)
from ..ml.trainer import LocalTrainer
from ..ml.vae import TelemetryVAE
from ..messaging.isl import (
    ISLBus,
    TOPIC_ALERTS,
    TOPIC_CRL_SYNC,
    TOPIC_FEDERATION_DELTAS,
    TOPIC_FEDERATION_GLOBAL,
    TOPIC_LSA,
)
from ..policy.pdp import PolicyConfig, PolicyDecisionPoint
from ..routing.link_state import LinkStateRouter

logger = logging.getLogger(__name__)

# Rotate certificates when 20% of TTL remains (60s before expiry at 300s TTL)
CERT_ROTATION_MARGIN_RATIO = 0.2


@dataclass
class NodeConfig:
    """Configuration for a satellite node."""

    node_id: str = "sat-01"
    nats_url: str = "nats://nats-server:4222"
    cert_dir: str = "/tmp/astraea/certs"
    is_aggregator: bool = False
    telemetry_hz: float = 10.0  # Telemetry ingestion rate
    federation_interval_s: float = 30.0  # Seconds between federation rounds
    lsa_interval_s: float = 5.0  # Seconds between LSA broadcasts
    training_epochs_per_round: int = 5
    training_batches_per_epoch: int = 10
    warmup_samples: int = 50
    enable_mtls: bool = True  # Enable mTLS on NATS ISL
    peers: List[str] = field(default_factory=list)
    peer_latencies: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> NodeConfig:
        """Load configuration from environment variables."""
        peers_str = os.environ.get("ASTRAEA_PEERS", "")
        peers = [p.strip() for p in peers_str.split(",") if p.strip()]

        latencies_str = os.environ.get("ASTRAEA_PEER_LATENCIES", "")
        latencies = {}
        for entry in latencies_str.split(","):
            entry = entry.strip()
            if ":" in entry:
                peer, lat = entry.split(":", 1)
                latencies[peer.strip()] = float(lat.strip())

        return cls(
            node_id=os.environ.get("ASTRAEA_NODE_ID", "sat-01"),
            nats_url=os.environ.get("ASTRAEA_NATS_URL", "nats://nats-server:4222"),
            cert_dir=os.environ.get("ASTRAEA_CERT_DIR", "/tmp/astraea/certs"),
            is_aggregator=os.environ.get("ASTRAEA_IS_AGGREGATOR", "").lower()
            in ("1", "true", "yes"),
            telemetry_hz=float(os.environ.get("ASTRAEA_TELEMETRY_HZ", "10.0")),
            federation_interval_s=float(
                os.environ.get("ASTRAEA_FEDERATION_INTERVAL", "30.0")
            ),
            enable_mtls=os.environ.get("ASTRAEA_ENABLE_MTLS", "true").lower()
            in ("1", "true", "yes"),
            peers=peers,
            peer_latencies=latencies,
        )


@dataclass
class HealthStatus:
    """Node health report for liveness/readiness probes."""

    node_id: str = ""
    online: bool = False
    uptime_s: float = 0.0
    cert_expires_in_s: float = 0.0
    anomaly_score: float = 0.0
    trust_level: str = "unknown"
    peers_reachable: int = 0
    federation_rounds: int = 0
    telemetry_ticks: int = 0


class SatelliteNode:
    """
    Core satellite node runtime.

    Orchestrates all subsystems and manages the asyncio event loop.
    """

    def __init__(self, config: NodeConfig, ca: Optional[CertificateAuthority] = None):
        self.config = config
        self.node_id = config.node_id

        # Crypto / Identity
        self.ca = ca or generate_ca()
        self.identity: Optional[NodeIdentity] = None
        self.crl = CRLCache()

        # ML / Anomaly Detection
        self.model = TelemetryVAE()
        self.detector = AnomalyDetector(
            self.model, warmup_samples=config.warmup_samples
        )
        self.trainer = LocalTrainer(self.model)

        # Federation
        self.aggregator: Optional[FederatedAggregator] = None
        if config.is_aggregator:
            self.aggregator = FederatedAggregator(self.model)

        # Routing
        self.router = LinkStateRouter(self.node_id)

        # Policy — register callbacks
        self.pdp = PolicyDecisionPoint(
            config=PolicyConfig(),
            on_revoke=self._handle_revocation,
            on_isolate=self._handle_isolation,
            on_trust_change=self._handle_trust_change,
        )

        # Messaging — will be initialized with mTLS in boot()
        self.isl: Optional[ISLBus] = None

        # Runtime state
        self._running = False
        self._time_step = 0
        self._boot_time = 0.0
        self._cert_issued_at = 0.0
        self._attack_mode = AttackMode.NOMINAL
        self._attack_intensity = 0.0
        self._tasks: List[asyncio.Task] = []

        # Peer certificate serial tracking for PDP
        self._peer_serials: Dict[str, int] = {}

    async def boot(self) -> None:
        """Full node bootstrap sequence."""
        logger.info("=" * 60)
        logger.info("ASTRAEA-1 NODE BOOT: %s", self.node_id)
        logger.info("=" * 60)

        # Phase 1: Identity Bootstrap — issue ephemeral certificate
        self._issue_certificate()

        # Register self with PDP
        self.pdp.register_node(
            self.node_id, self.identity.certificate.serial_number
        )

        # Pre-register known peers with PDP (serial=0 as placeholder until
        # we receive their actual cert serial via LSA/alert)
        for peer in self.config.peers:
            self.pdp.register_node(peer, cert_serial=0)

        # Phase 2: Initialize routing topology
        for peer in self.config.peers:
            latency = self.config.peer_latencies.get(peer, 400.0)
            self.router.add_link(self.node_id, peer, latency)
            self.router.add_link(peer, self.node_id, latency)
        self.router.recompute_routes()

        # Phase 3: Initialize local trainer
        self.trainer.snapshot_global_weights()

        # Phase 4: Pre-train on nominal data (warmup)
        logger.info("Pre-training on nominal telemetry (warmup)...")
        for _ in range(3):
            self.trainer.train_epoch(batch_size=64, num_batches=20)

        # Feed warmup samples to detector
        for i in range(self.config.warmup_samples + 10):
            raw = generate_telemetry_batch(1, AttackMode.NOMINAL, time_step=i)
            x = normalize_telemetry(raw)
            self.detector.ingest(x.squeeze(0))

        logger.info(
            "Warmup complete: detector_threshold=%.4f",
            self.detector.state.threshold,
        )

        # Phase 5: Initialize ISL bus with mTLS context
        tls_ctx = None
        if self.config.enable_mtls and self.identity is not None:
            try:
                tls_ctx = create_mtls_client_context(self.identity)
                logger.info("mTLS context created for ISL bus")
            except Exception:
                logger.warning("Failed to create mTLS context — ISL will run without TLS")

        self.isl = ISLBus(
            node_id=self.node_id,
            nats_url=self.config.nats_url,
            tls_context=tls_ctx,
        )
        await self.isl.connect()
        await self._setup_subscriptions()

        self._boot_time = time.time()
        self._running = True
        logger.info("Node %s ONLINE (mTLS=%s)", self.node_id, tls_ctx is not None)

    async def run(self) -> None:
        """Start all concurrent event loops."""
        await self.boot()

        self._tasks = [
            asyncio.create_task(self._telemetry_loop()),
            asyncio.create_task(self._federation_loop()),
            asyncio.create_task(self._routing_loop()),
            asyncio.create_task(self._cert_rotation_loop()),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Node %s shutting down...", self.node_id)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self.isl:
            await self.isl.close()
        logger.info("Node %s OFFLINE", self.node_id)

    # ------------------------------------------------------------------
    # Certificate Management
    # ------------------------------------------------------------------

    def _issue_certificate(self) -> None:
        """Issue (or re-issue) an ephemeral X.509 certificate."""
        self.identity = issue_node_certificate(self.ca, self.node_id)
        cert_dir = Path(self.config.cert_dir) / self.node_id
        persist_identity(self.identity, cert_dir)
        self._cert_issued_at = time.time()
        logger.info(
            "Identity: %s  serial=%d  ttl=%ds",
            self.identity.spiffe_id,
            self.identity.certificate.serial_number,
            CERT_TTL_SECONDS,
        )

    async def _cert_rotation_loop(self) -> None:
        """
        Periodically rotate the ephemeral certificate before TTL expiry.

        Rotation happens when the remaining lifetime drops below
        CERT_ROTATION_MARGIN_RATIO * CERT_TTL_SECONDS (default: 60s).
        """
        margin = CERT_TTL_SECONDS * CERT_ROTATION_MARGIN_RATIO
        check_interval = max(margin / 2, 10.0)
        logger.info(
            "Cert rotation loop started (ttl=%ds  margin=%.0fs  check=%.0fs)",
            CERT_TTL_SECONDS,
            margin,
            check_interval,
        )

        while self._running:
            await asyncio.sleep(check_interval)
            if not self._running:
                break

            elapsed = time.time() - self._cert_issued_at
            remaining = CERT_TTL_SECONDS - elapsed

            if remaining <= margin:
                logger.info(
                    "Certificate rotation: %.0fs remaining (margin=%.0fs) — re-issuing",
                    remaining,
                    margin,
                )
                old_serial = self.identity.certificate.serial_number
                self._issue_certificate()

                # Update mTLS context on ISL if enabled
                if self.config.enable_mtls and self.isl is not None:
                    try:
                        new_ctx = create_mtls_client_context(self.identity)
                        self.isl.tls_context = new_ctx
                        logger.info(
                            "mTLS context rotated: old_serial=%d  new_serial=%d",
                            old_serial,
                            self.identity.certificate.serial_number,
                        )
                    except Exception:
                        logger.warning("Failed to rotate mTLS context")

    # ------------------------------------------------------------------
    # Core Event Loops
    # ------------------------------------------------------------------

    async def _telemetry_loop(self) -> None:
        """
        Main telemetry ingestion and anomaly detection loop.

        Runs at config.telemetry_hz and processes one sample per tick.
        """
        interval = 1.0 / self.config.telemetry_hz
        logger.info("Telemetry loop started @ %.1f Hz", self.config.telemetry_hz)

        while self._running:
            raw = generate_telemetry_batch(
                1,
                attack_mode=self._attack_mode,
                time_step=self._time_step,
                attack_intensity=self._attack_intensity,
            )
            x = normalize_telemetry(raw).squeeze(0)

            # VAE inference + anomaly detection
            state = self.detector.ingest(x)
            self._time_step += 1

            # Policy evaluation
            self.pdp.evaluate(self.node_id, state.last_normalized_score)

            if state.is_anomalous and self._time_step % 10 == 0:
                alert = {
                    "type": "anomaly_alert",
                    "node_id": self.node_id,
                    "cert_serial": self.identity.certificate.serial_number,
                    "anomaly_score": state.last_normalized_score,
                    "raw_score": state.last_score,
                    "threshold": state.threshold,
                    "timestamp": time.time(),
                }
                await self.isl.publish_json(TOPIC_ALERTS, alert)

            await asyncio.sleep(interval)

    async def _federation_loop(self) -> None:
        """
        Periodic federation round: train locally, submit delta, apply global.
        """
        interval = self.config.federation_interval_s
        logger.info("Federation loop started (interval=%.1fs)", interval)

        while self._running:
            await asyncio.sleep(interval)

            if not self._running:
                break

            for _ in range(self.config.training_epochs_per_round):
                self.trainer.train_epoch(
                    batch_size=64,
                    num_batches=self.config.training_batches_per_epoch,
                    time_step=self._time_step,
                )

            delta = self.trainer.compute_weight_delta()

            buf = io.BytesIO()
            torch.save(
                {"node_id": self.node_id, "delta": delta},
                buf,
            )
            await self.isl.publish(TOPIC_FEDERATION_DELTAS, buf.getvalue())

            if self.aggregator:
                self.aggregator.submit_delta(
                    self.node_id,
                    delta,
                    revoked_serials=self.crl.get_all_revoked(),
                )

            logger.info("Federation delta submitted from %s", self.node_id)

    async def _routing_loop(self) -> None:
        """Periodic Link-State Advertisement broadcast."""
        interval = self.config.lsa_interval_s
        logger.info("Routing LSA loop started (interval=%.1fs)", interval)

        while self._running:
            await asyncio.sleep(interval)

            if not self._running:
                break

            lsa = self.router.generate_lsa()
            # Include our cert serial so peers can register us in their PDP
            lsa["cert_serial"] = self.identity.certificate.serial_number
            lsa["spiffe_id"] = self.identity.spiffe_id
            await self.isl.publish_json(TOPIC_LSA, lsa)

    # ------------------------------------------------------------------
    # Message Handlers (ISL Subscriptions)
    # ------------------------------------------------------------------

    async def _setup_subscriptions(self) -> None:
        """Set up ISL topic subscriptions."""
        await self.isl.subscribe(TOPIC_CRL_SYNC, self._on_crl_sync)
        await self.isl.subscribe(TOPIC_LSA, self._on_lsa)
        await self.isl.subscribe(TOPIC_ALERTS, self._on_alert)
        await self.isl.subscribe(
            TOPIC_FEDERATION_GLOBAL, self._on_global_model
        )
        if self.aggregator:
            await self.isl.subscribe(
                TOPIC_FEDERATION_DELTAS, self._on_federation_delta
            )

    async def _on_crl_sync(self, msg) -> None:
        """Handle CRL synchronization broadcast from peers."""
        try:
            revoked = CRLCache.deserialize_broadcast(msg.data)
            added = self.crl.merge_from_broadcast(revoked, source="peer")
            if added > 0:
                logger.info("CRL sync: merged %d new revocations", added)
        except Exception:
            logger.exception("Error processing CRL sync")

    async def _on_lsa(self, msg) -> None:
        """Handle Link-State Advertisement from a peer."""
        try:
            lsa = json.loads(msg.data.decode())
            origin = lsa.get("origin")
            if origin == self.node_id:
                return

            # Register peer's certificate serial with PDP if we haven't yet
            cert_serial = lsa.get("cert_serial")
            if origin and cert_serial and origin not in self._peer_serials:
                self._peer_serials[origin] = cert_serial
                # Update PDP registration with actual serial
                pdp_state = self.pdp.get_node_state(origin)
                if pdp_state is not None and pdp_state.cert_serial == 0:
                    pdp_state.cert_serial = cert_serial
                    logger.info(
                        "Registered peer cert: node=%s  serial=%d",
                        origin,
                        cert_serial,
                    )
                elif pdp_state is None:
                    self.pdp.register_node(origin, cert_serial)

            # Validate peer SPIFFE ID format
            spiffe_id = lsa.get("spiffe_id", "")
            if spiffe_id and not spiffe_id.startswith("spiffe://astraea-1.mesh/"):
                logger.warning(
                    "Rejected LSA from %s: invalid SPIFFE ID %s",
                    origin,
                    spiffe_id,
                )
                return

            self.router.apply_lsa(lsa)
        except Exception:
            logger.exception("Error processing LSA")

    async def _on_alert(self, msg) -> None:
        """Handle anomaly alert from a peer."""
        try:
            alert = json.loads(msg.data.decode())
            source = alert.get("node_id", "unknown")
            if source == self.node_id:
                return

            score = alert.get("anomaly_score", 0)

            # Register peer cert serial if included in alert
            cert_serial = alert.get("cert_serial")
            if cert_serial and source not in self._peer_serials:
                self._peer_serials[source] = cert_serial
                pdp_state = self.pdp.get_node_state(source)
                if pdp_state is not None and pdp_state.cert_serial == 0:
                    pdp_state.cert_serial = cert_serial
                elif pdp_state is None:
                    self.pdp.register_node(source, cert_serial)

            self.pdp.evaluate(source, score)
            logger.warning(
                "Peer anomaly alert: node=%s  score=%.4f", source, score
            )
        except Exception:
            logger.exception("Error processing alert")

    async def _on_global_model(self, msg) -> None:
        """Handle global model broadcast from the aggregator."""
        try:
            global_state = FederatedAggregator.deserialize_global_state(
                msg.data
            )
            self.trainer.apply_global_weights(global_state)
            self.detector.reset_ema()
            logger.info("Applied global model update")
        except Exception:
            logger.exception("Error applying global model")

    async def _on_federation_delta(self, msg) -> None:
        """Handle weight delta submission (aggregator only)."""
        if not self.aggregator:
            return
        try:
            buf = io.BytesIO(msg.data)
            payload = torch.load(buf, weights_only=False)
            node_id = payload["node_id"]
            delta = payload["delta"]

            accepted = self.aggregator.submit_delta(
                node_id,
                delta,
                revoked_serials=self.crl.get_all_revoked(),
            )

            if accepted:
                result = self.aggregator.aggregate(
                    excluded_nodes=set(self.pdp.get_revoked_nodes())
                )
                if result is not None:
                    data = self.aggregator.serialize_global_state()
                    await self.isl.publish(TOPIC_FEDERATION_GLOBAL, data)
                    logger.info("Global model broadcast after aggregation")
        except Exception:
            logger.exception("Error processing federation delta")

    # ------------------------------------------------------------------
    # Policy Callbacks
    # ------------------------------------------------------------------

    def _handle_revocation(self, node_id: str, cert_serial: int) -> None:
        """Callback: PDP has decided to revoke a node's certificate."""
        logger.critical(
            "REVOCATION: node=%s  serial=%d", node_id, cert_serial
        )
        self.crl.add_revocation(cert_serial, source="pdp")

        # Broadcast CRL update to constellation
        if self.isl:
            asyncio.ensure_future(
                self.isl.publish(TOPIC_CRL_SYNC, self.crl.serialize())
            )

    def _handle_isolation(self, node_id: str) -> None:
        """Callback: PDP requires mesh isolation of a node."""
        logger.critical("ISOLATION: removing %s from routing mesh", node_id)
        self.router.isolate_node(node_id)

    def _handle_trust_change(self, node_id: str, new_trust: float) -> None:
        """Callback: trust score changed — update routing weights."""
        self.router.update_trust_score(node_id, new_trust)

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Return current health status for liveness probes."""
        cert_remaining = 0.0
        if self._cert_issued_at > 0:
            cert_remaining = max(
                0.0, CERT_TTL_SECONDS - (time.time() - self._cert_issued_at)
            )

        pdp_state = self.pdp.get_node_state(self.node_id)

        return HealthStatus(
            node_id=self.node_id,
            online=self._running,
            uptime_s=time.time() - self._boot_time if self._boot_time else 0.0,
            cert_expires_in_s=cert_remaining,
            anomaly_score=self.detector.get_anomaly_score(),
            trust_level=pdp_state.trust_level.value if pdp_state else "unknown",
            peers_reachable=len(self.router.get_reachable_nodes()),
            federation_rounds=self.aggregator.round_id if self.aggregator else 0,
            telemetry_ticks=self._time_step,
        )

    # ------------------------------------------------------------------
    # Attack Injection (for Red Team testing)
    # ------------------------------------------------------------------

    def inject_attack(
        self, mode: AttackMode, intensity: float = 1.0
    ) -> None:
        """Inject an attack pattern into this node's telemetry stream."""
        logger.warning(
            "ATTACK INJECTED: node=%s  mode=%s  intensity=%.2f",
            self.node_id,
            mode.value,
            intensity,
        )
        self._attack_mode = mode
        self._attack_intensity = intensity

    def stop_attack(self) -> None:
        """Revert to nominal telemetry."""
        self._attack_mode = AttackMode.NOMINAL
        self._attack_intensity = 0.0


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for running a satellite node."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config = NodeConfig.from_env()
    logger.info("Loaded config for node %s", config.node_id)

    node = SatelliteNode(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        logger.info("Received shutdown signal")
        loop.create_task(node.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(node.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
