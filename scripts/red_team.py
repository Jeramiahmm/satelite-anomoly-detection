"""
Red-Team Attack Script — "Polarized Telemetry Injection"
=========================================================

Simulates a sophisticated adversary performing a multi-phase attack
against the Astraea-1 constellation:

    Phase 1: Reconnaissance  (5s)  — Observe nominal telemetry baseline
    Phase 2: Injection       (var) — Inject polarized telemetry drift
    Phase 3: Observation     (5s)  — Verify detection & isolation occurred

Attack vector: "Polarized Telemetry Injection"
    A gradual drift attack where compromised sensor readings are slowly
    shifted away from nominal values. This is harder to detect than
    sudden spikes (bitflip) because it mimics natural orbital variation.

Success criteria:
    - The system must detect the anomaly
    - The PDP must revoke the node's certificate
    - The routing mesh must isolate the node
    - Total detection-to-isolation time < 2 seconds

Usage:
    # Standalone (no Docker):
    python -m scripts.red_team

    # Inside Docker:
    docker compose exec sat-03 python -m scripts.red_team

    # Target a specific node:
    python -m scripts.red_team --target sat-03 --attack spacejack
"""

from __future__ import annotations

import argparse
import asyncio
import json as json_module
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.crypto.identity import generate_ca, issue_node_certificate
from astraea.crypto.crl_manager import CRLCache
from astraea.federation.aggregator import FederatedAggregator
from astraea.ml.anomaly_detector import AnomalyDetector
from astraea.ml.telemetry import (
    AttackMode,
    generate_telemetry_batch,
    normalize_telemetry,
)
from astraea.ml.trainer import LocalTrainer
from astraea.ml.vae import TelemetryVAE
from astraea.node.satellite_node import NodeConfig, SatelliteNode
from astraea.policy.pdp import PolicyConfig, PolicyDecisionPoint, TrustLevel
from astraea.routing.link_state import LinkStateRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("red_team")


class RedTeamExercise:
    """
    Orchestrates a red-team attack simulation against the constellation.

    This runs a self-contained simulation with multiple nodes, a shared
    CA, and the full detection pipeline — no Docker required.
    """

    def __init__(
        self,
        target_node: str = "sat-03",
        attack_mode: AttackMode = AttackMode.POLARIZED,
        attack_intensity: float = 1.0,
        num_nodes: int = 5,
        detection_sla_s: float = 2.0,
    ):
        self.target = target_node
        self.attack_mode = attack_mode
        self.attack_intensity = attack_intensity
        self.num_nodes = num_nodes
        self.detection_sla = detection_sla_s

        # Shared constellation CA
        self.ca = generate_ca()
        self.nodes: dict[str, dict] = {}
        self.crl = CRLCache()

        # Timing
        self.attack_start_time = 0.0
        self.detection_time = 0.0
        self.isolation_time = 0.0
        self.revocation_time = 0.0

    def setup(self) -> None:
        """Initialize all nodes and the constellation topology."""
        logger.info("=" * 70)
        logger.info("  RED TEAM EXERCISE: Polarized Telemetry Injection")
        logger.info("=" * 70)
        logger.info("  Target node:    %s", self.target)
        logger.info("  Attack mode:    %s", self.attack_mode.value)
        logger.info("  Intensity:      %.2f", self.attack_intensity)
        logger.info("  Detection SLA:  < %.1fs", self.detection_sla)
        logger.info("=" * 70)

        node_ids = [f"sat-{i:02d}" for i in range(1, self.num_nodes + 1)]

        for nid in node_ids:
            identity = issue_node_certificate(self.ca, nid)
            model = TelemetryVAE()
            detector = AnomalyDetector(model, warmup_samples=50)
            trainer = LocalTrainer(model)
            trainer.snapshot_global_weights()

            router = LinkStateRouter(nid)
            pdp = PolicyDecisionPoint(
                config=PolicyConfig(consecutive_required=2),
            )
            pdp.register_node(nid, identity.certificate.serial_number)

            self.nodes[nid] = {
                "identity": identity,
                "model": model,
                "detector": detector,
                "trainer": trainer,
                "router": router,
                "pdp": pdp,
                "serial": identity.certificate.serial_number,
            }

        # Build ring topology: sat-01 → sat-02 → ... → sat-05 → sat-01
        for i, nid in enumerate(node_ids):
            next_nid = node_ids[(i + 1) % len(node_ids)]
            for n in self.nodes.values():
                n["router"].add_link(nid, next_nid, base_latency_ms=300.0)
                n["router"].add_link(next_nid, nid, base_latency_ms=300.0)

        for n in self.nodes.values():
            n["router"].recompute_routes()

        logger.info("Constellation initialized: %d nodes in ring topology", len(node_ids))

    def pretrain(self, epochs: int = 5) -> None:
        """Pre-train all nodes on nominal telemetry."""
        logger.info("\n[Phase 0] Pre-training all nodes on nominal data...")

        for nid, n in self.nodes.items():
            for _ in range(epochs):
                n["trainer"].train_epoch(
                    batch_size=64, num_batches=20, time_step=0
                )

            # Warm up detector
            for t in range(60):
                raw = generate_telemetry_batch(1, AttackMode.NOMINAL, time_step=t)
                x = normalize_telemetry(raw).squeeze(0)
                n["detector"].ingest(x)

            logger.info(
                "  %s pre-trained  threshold=%.4f",
                nid,
                n["detector"].state.threshold,
            )

    def run_attack(self) -> dict:
        """
        Execute the full attack sequence and measure response.

        Returns a dict with timing measurements and pass/fail status.
        """
        self.setup()
        self.pretrain()

        target_node = self.nodes[self.target]
        target_pdp = target_node["pdp"]
        target_detector = target_node["detector"]
        target_serial = target_node["serial"]

        # Phase 1: Reconnaissance — confirm nominal baseline
        logger.info("\n[Phase 1] Reconnaissance — confirming nominal baseline...")
        for t in range(50):
            raw = generate_telemetry_batch(
                1, AttackMode.NOMINAL, time_step=100 + t
            )
            x = normalize_telemetry(raw).squeeze(0)
            state = target_detector.ingest(x)

        logger.info(
            "  Baseline confirmed: score=%.4f  threshold=%.4f  anomalous=%s",
            state.last_normalized_score,
            state.threshold,
            state.is_anomalous,
        )
        assert not state.is_anomalous, "Baseline should not be anomalous!"

        # Phase 2: Attack injection
        logger.info("\n[Phase 2] ATTACK — Injecting %s telemetry...", self.attack_mode.value)
        self.attack_start_time = time.time()

        detected = False
        revoked = False
        max_ticks = 500  # Safety limit

        for t in range(max_ticks):
            raw = generate_telemetry_batch(
                1,
                attack_mode=self.attack_mode,
                time_step=200 + t,
                attack_intensity=self.attack_intensity,
            )
            x = normalize_telemetry(raw).squeeze(0)
            state = target_detector.ingest(x)

            # Evaluate policy
            pdp_state = target_pdp.evaluate(
                self.target, state.last_normalized_score
            )

            if state.is_anomalous and not detected:
                detected = True
                self.detection_time = time.time()
                logger.warning(
                    "  *** ANOMALY DETECTED at tick %d ***  "
                    "score=%.4f  threshold=%.4f  elapsed=%.3fs",
                    t,
                    state.last_normalized_score,
                    state.threshold,
                    self.detection_time - self.attack_start_time,
                )

            if pdp_state.revoked and not revoked:
                revoked = True
                self.revocation_time = time.time()

                # Simulate CRL propagation
                self.crl.add_revocation(target_serial, source="pdp")

                # Isolate in all routers
                for nid, n in self.nodes.items():
                    n["router"].isolate_node(self.target)

                self.isolation_time = time.time()

                logger.critical(
                    "  *** NODE REVOKED & ISOLATED at tick %d ***  "
                    "serial=%d  elapsed=%.3fs",
                    t,
                    target_serial,
                    self.isolation_time - self.attack_start_time,
                )
                break

        # Phase 3: Post-attack verification
        logger.info("\n[Phase 3] Post-attack verification...")

        # Verify CRL
        is_revoked = self.crl.is_revoked(target_serial)
        logger.info("  CRL check: serial=%d revoked=%s", target_serial, is_revoked)

        # Verify routing isolation
        isolated_count = 0
        for nid, n in self.nodes.items():
            if nid == self.target:
                continue
            reachable = n["router"].get_reachable_nodes()
            if self.target not in reachable:
                isolated_count += 1

        total_peers = len(self.nodes) - 1
        logger.info(
            "  Routing isolation: %d/%d peers cannot reach %s",
            isolated_count,
            total_peers,
            self.target,
        )

        # Calculate timings
        total_elapsed = (
            (self.isolation_time - self.attack_start_time)
            if self.isolation_time > 0
            else float("inf")
        )
        detection_elapsed = (
            (self.detection_time - self.attack_start_time)
            if self.detection_time > 0
            else float("inf")
        )

        passed = (
            detected
            and revoked
            and is_revoked
            and isolated_count == total_peers
            and total_elapsed < self.detection_sla
        )

        results = {
            "target": self.target,
            "attack_mode": self.attack_mode.value,
            "intensity": self.attack_intensity,
            "detected": detected,
            "revoked": revoked,
            "crl_revoked": is_revoked,
            "isolated_peers": isolated_count,
            "total_peers": total_peers,
            "detection_time_s": round(detection_elapsed, 4),
            "total_time_s": round(total_elapsed, 4),
            "sla_target_s": self.detection_sla,
            "sla_met": total_elapsed < self.detection_sla,
            "passed": passed,
        }

        # Print report
        logger.info("\n" + "=" * 70)
        logger.info("  RED TEAM EXERCISE — RESULTS")
        logger.info("=" * 70)
        logger.info("  Attack Mode:         %s", results["attack_mode"])
        logger.info("  Anomaly Detected:    %s", results["detected"])
        logger.info("  Certificate Revoked: %s", results["revoked"])
        logger.info("  CRL Updated:         %s", results["crl_revoked"])
        logger.info(
            "  Mesh Isolated:       %d/%d peers",
            results["isolated_peers"],
            results["total_peers"],
        )
        logger.info("  Detection Time:      %.4fs", results["detection_time_s"])
        logger.info("  Total D→I Time:      %.4fs", results["total_time_s"])
        logger.info("  SLA Target:          < %.1fs", results["sla_target_s"])
        logger.info("  SLA Met:             %s", results["sla_met"])
        logger.info("-" * 70)

        if passed:
            logger.info("  RESULT: *** PASS *** — System detected and isolated "
                       "the compromised node within SLA.")
        else:
            logger.error("  RESULT: *** FAIL *** — System did not meet "
                        "detection/isolation requirements.")

        logger.info("=" * 70)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astraea-1 Red Team Attack Simulation"
    )
    parser.add_argument(
        "--target",
        default="sat-03",
        help="Target node ID (default: sat-03)",
    )
    parser.add_argument(
        "--attack",
        choices=["polarized", "spacejack", "bitflip"],
        default="polarized",
        help="Attack type (default: polarized)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="Attack intensity 0.0–1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of constellation nodes (default: 5)",
    )
    parser.add_argument(
        "--sla",
        type=float,
        default=2.0,
        help="Detection-to-isolation SLA in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write JSON results file (e.g., results/polarized.json)",
    )

    args = parser.parse_args()

    attack_map = {
        "polarized": AttackMode.POLARIZED,
        "spacejack": AttackMode.SPACEJACK,
        "bitflip": AttackMode.BITFLIP,
    }

    exercise = RedTeamExercise(
        target_node=args.target,
        attack_mode=attack_map[args.attack],
        attack_intensity=args.intensity,
        num_nodes=args.nodes,
        detection_sla_s=args.sla,
    )

    results = exercise.run_attack()

    # Export results to JSON file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Replace infinity with null for valid JSON
        sanitized = {
            k: (None if isinstance(v, float) and v == float("inf") else v)
            for k, v in results.items()
        }
        output_path.write_text(json_module.dumps(sanitized, indent=2) + "\n")
        logger.info("Results written to %s", output_path)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
