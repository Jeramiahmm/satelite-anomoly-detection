"""
Policy Decision Point (PDP) for Astraea-1 Zero-Trust Mesh.

Evaluates anomaly scores against configurable policies and triggers
certificate revocation + mesh isolation when thresholds are exceeded.

This is the enforcement layer of the NIST SP 800-207 Zero Trust Architecture:
    - Policy Decision Point (PDP): This module
    - Policy Enforcement Point (PEP): The mTLS layer + CRL cache
    - Policy Information Point (PIP): The VAE anomaly detector

Policy rules:
    1. If anomaly_score > REVOCATION_THRESHOLD → revoke certificate
    2. If anomaly_score > ISOLATION_THRESHOLD → isolate from routing mesh
    3. Consecutive anomaly threshold before action (debounce)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust classification per NIST SP 800-207 §2."""

    TRUSTED = "trusted"
    DEGRADED = "degraded"
    UNTRUSTED = "untrusted"
    REVOKED = "revoked"


@dataclass
class NodeSecurityState:
    """Security posture of a single node as seen by the PDP."""

    node_id: str
    cert_serial: Optional[int] = None
    trust_level: TrustLevel = TrustLevel.TRUSTED
    trust_score: float = 1.0
    anomaly_score: float = 0.0
    consecutive_anomalies: int = 0
    last_evaluation: float = 0.0
    revoked: bool = False
    revocation_time: float = 0.0
    isolation_time: float = 0.0


@dataclass
class PolicyConfig:
    """Tunable policy parameters."""

    revocation_threshold: float = 0.8  # Anomaly score → revoke cert
    isolation_threshold: float = 0.6   # Anomaly score → degrade routes
    degraded_threshold: float = 0.4    # Anomaly score → lower trust
    consecutive_required: int = 2      # Debounce: N consecutive anomalies
    trust_sensitivity: float = 1.25    # α in trust_score = max(0, 1 - α*anomaly)
    recovery_enabled: bool = False     # Allow automatic trust restoration


class PolicyDecisionPoint:
    """
    Central PDP evaluating node security posture.

    On each evaluation cycle:
        1. Receive anomaly score from the anomaly detector.
        2. Apply trust classification rules.
        3. If REVOKED → trigger CRL update + routing isolation.
    """

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        on_revoke: Optional[Callable[[str, int], None]] = None,
        on_isolate: Optional[Callable[[str], None]] = None,
        on_trust_change: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Args:
            config: Policy tuning parameters.
            on_revoke: Callback(node_id, cert_serial) on revocation.
            on_isolate: Callback(node_id) on routing isolation.
            on_trust_change: Callback(node_id, new_trust_score) on change.
        """
        self.config = config or PolicyConfig()
        self._on_revoke = on_revoke
        self._on_isolate = on_isolate
        self._on_trust_change = on_trust_change
        self._nodes: Dict[str, NodeSecurityState] = {}
        self._actions_log: List[Dict] = []

    def register_node(self, node_id: str, cert_serial: int) -> None:
        """Register a node with the PDP."""
        self._nodes[node_id] = NodeSecurityState(
            node_id=node_id,
            cert_serial=cert_serial,
        )
        logger.info("PDP registered node %s (serial=%d)", node_id, cert_serial)

    def evaluate(self, node_id: str, anomaly_score: float) -> NodeSecurityState:
        """
        Evaluate a node's anomaly score and apply trust policy.

        This is the core PDP decision loop:
            score > 0.8 → REVOKE (cert invalidation + mesh isolation)
            score > 0.6 → UNTRUSTED (routing weight penalty)
            score > 0.4 → DEGRADED (monitoring intensified)
            else → TRUSTED

        Returns:
            Updated NodeSecurityState.
        """
        state = self._nodes.get(node_id)
        if state is None:
            logger.error("Unknown node %s — cannot evaluate", node_id)
            state = NodeSecurityState(node_id=node_id)
            self._nodes[node_id] = state

        # Already revoked — no re-evaluation (unless recovery is enabled)
        if state.revoked and not self.config.recovery_enabled:
            return state

        state.anomaly_score = anomaly_score
        state.last_evaluation = time.time()

        # Compute trust score: trust = max(0, 1 - α * anomaly)
        new_trust = max(0.0, 1.0 - self.config.trust_sensitivity * anomaly_score)
        old_trust = state.trust_score
        state.trust_score = new_trust

        # Classify trust level
        cfg = self.config
        old_level = state.trust_level

        if anomaly_score >= cfg.revocation_threshold:
            state.consecutive_anomalies += 1
        else:
            state.consecutive_anomalies = 0

        if (
            anomaly_score >= cfg.revocation_threshold
            and state.consecutive_anomalies >= cfg.consecutive_required
        ):
            state.trust_level = TrustLevel.REVOKED
        elif anomaly_score >= cfg.isolation_threshold:
            state.trust_level = TrustLevel.UNTRUSTED
        elif anomaly_score >= cfg.degraded_threshold:
            state.trust_level = TrustLevel.DEGRADED
        else:
            state.trust_level = TrustLevel.TRUSTED

        # Fire callbacks on state transitions
        if abs(new_trust - old_trust) > 0.01 and self._on_trust_change:
            self._on_trust_change(node_id, new_trust)

        if state.trust_level == TrustLevel.REVOKED and not state.revoked:
            state.revoked = True
            state.revocation_time = time.time()
            state.trust_score = 0.0

            action = {
                "action": "REVOKE",
                "node_id": node_id,
                "cert_serial": state.cert_serial,
                "anomaly_score": anomaly_score,
                "timestamp": state.revocation_time,
            }
            self._actions_log.append(action)

            logger.critical(
                "PDP DECISION: REVOKE node=%s  serial=%d  score=%.4f  "
                "consecutive=%d",
                node_id,
                state.cert_serial or -1,
                anomaly_score,
                state.consecutive_anomalies,
            )

            if self._on_revoke and state.cert_serial is not None:
                self._on_revoke(node_id, state.cert_serial)

        if (
            state.trust_level in (TrustLevel.UNTRUSTED, TrustLevel.REVOKED)
            and old_level not in (TrustLevel.UNTRUSTED, TrustLevel.REVOKED)
        ):
            state.isolation_time = time.time()
            if self._on_isolate:
                self._on_isolate(node_id)

        return state

    def get_node_state(self, node_id: str) -> Optional[NodeSecurityState]:
        return self._nodes.get(node_id)

    def get_trusted_nodes(self) -> List[str]:
        """Return list of node IDs that are not revoked."""
        return [
            nid
            for nid, state in self._nodes.items()
            if not state.revoked
        ]

    def get_revoked_nodes(self) -> List[str]:
        return [
            nid
            for nid, state in self._nodes.items()
            if state.revoked
        ]

    def get_actions_log(self) -> List[Dict]:
        return list(self._actions_log)
