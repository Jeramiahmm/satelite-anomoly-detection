"""
Prometheus-compatible metrics and HTTP health server for Astraea-1.

Exposes node telemetry, anomaly detection state, trust scores, and
federation status as Prometheus text-format metrics on port 9090.

Endpoints:
    GET /metrics  — Prometheus scrape target
    GET /health   — JSON health check (for Docker / Kubernetes probes)
    GET /status   — Human-readable node status page

No external dependencies — uses only the Python stdlib http.server.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..node.satellite_node import SatelliteNode

logger = logging.getLogger(__name__)

# Default port (matches EXPOSE 9090 in Dockerfile)
METRICS_PORT = 9090


class MetricsCollector:
    """
    Collects metrics from a SatelliteNode and formats them for Prometheus.

    Thread-safe: the HTTP server runs in a daemon thread while the node
    runs in the asyncio event loop on the main thread.
    """

    def __init__(self, node: "SatelliteNode"):
        self._node = node
        self._lock = threading.Lock()
        self._custom_counters: Dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._custom_counters[name] = self._custom_counters.get(name, 0) + value

    def collect_prometheus(self) -> str:
        """Generate Prometheus text-format metrics."""
        lines: List[str] = []
        node = self._node
        node_id = node.node_id

        def _gauge(name: str, help_text: str, value: float, labels: str = "") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lbl = f'{{node="{node_id}"{labels}}}' if labels else f'{{node="{node_id}"}}'
            lines.append(f"{name}{lbl} {value}")

        def _counter(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f'{name}{{node="{node_id}"}} {value}')

        # -- Node status --
        _gauge("astraea_node_online", "Whether the node is online (1/0)",
               1.0 if node._running else 0.0)

        uptime = time.time() - node._boot_time if node._boot_time else 0.0
        _gauge("astraea_node_uptime_seconds", "Node uptime in seconds", uptime)

        # -- Certificate --
        cert_remaining = 0.0
        if node._cert_issued_at > 0:
            from ..crypto.identity import CERT_TTL_SECONDS
            cert_remaining = max(0.0, CERT_TTL_SECONDS - (time.time() - node._cert_issued_at))
        _gauge("astraea_cert_ttl_remaining_seconds",
               "Seconds until current certificate expires", cert_remaining)

        # -- Anomaly detection --
        detector = node.detector
        _gauge("astraea_anomaly_score",
               "Current normalized anomaly score [0-1]",
               detector.state.last_normalized_score)
        _gauge("astraea_anomaly_raw_score",
               "Raw VAE reconstruction error",
               detector.state.last_score)
        _gauge("astraea_anomaly_threshold",
               "Dynamic anomaly detection threshold",
               detector.state.threshold)
        _gauge("astraea_anomaly_ema_mean",
               "EMA mean of reconstruction error",
               detector.state.ema_mean)
        _counter("astraea_anomaly_detections_total",
                 "Total number of anomaly detections",
                 detector.state.anomaly_count)
        _counter("astraea_telemetry_samples_total",
                 "Total telemetry samples processed",
                 detector.state.sample_count)

        # -- Trust / PDP --
        pdp_state = node.pdp.get_node_state(node_id)
        if pdp_state:
            _gauge("astraea_trust_score",
                   "Current trust score [0-1]",
                   pdp_state.trust_score)
            trust_level_map = {"trusted": 3, "degraded": 2, "untrusted": 1, "revoked": 0}
            _gauge("astraea_trust_level",
                   "Trust level (3=trusted, 2=degraded, 1=untrusted, 0=revoked)",
                   trust_level_map.get(pdp_state.trust_level.value, -1))

        # Per-peer trust
        for peer_id in node.config.peers:
            peer_state = node.pdp.get_node_state(peer_id)
            if peer_state:
                lbl = f',peer="{peer_id}"'
                _gauge("astraea_peer_trust_score",
                       "Trust score for a peer node",
                       peer_state.trust_score, labels=lbl)

        # -- Routing --
        reachable = node.router.get_reachable_nodes()
        _gauge("astraea_routing_reachable_peers",
               "Number of peers reachable via routing",
               len(reachable))

        # -- Federation --
        _counter("astraea_telemetry_ticks_total",
                 "Total telemetry processing ticks",
                 node._time_step)
        if node.aggregator:
            _gauge("astraea_federation_round",
                   "Current federation round number",
                   node.aggregator.round_id)

        # -- CRL --
        revoked_count = len(node.crl.get_all_revoked())
        _gauge("astraea_crl_revoked_count",
               "Number of revoked certificates in local CRL",
               revoked_count)

        # Custom counters
        with self._lock:
            for name, value in self._custom_counters.items():
                _counter(f"astraea_{name}", f"Custom counter: {name}", value)

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def collect_health_json(self) -> dict:
        """Generate a JSON health check response."""
        h = self._node.health()
        return {
            "node_id": h.node_id,
            "online": h.online,
            "uptime_s": round(h.uptime_s, 1),
            "cert_expires_in_s": round(h.cert_expires_in_s, 1),
            "anomaly_score": round(h.anomaly_score, 4),
            "trust_level": h.trust_level,
            "peers_reachable": h.peers_reachable,
            "federation_rounds": h.federation_rounds,
            "telemetry_ticks": h.telemetry_ticks,
        }

    def collect_status_text(self) -> str:
        """Generate a human-readable status page."""
        h = self._node.health()
        state = self._node.detector.state
        lines = [
            f"ASTRAEA-1 NODE STATUS: {h.node_id}",
            "=" * 50,
            f"  Online:           {h.online}",
            f"  Uptime:           {h.uptime_s:.0f}s",
            f"  Cert Expires In:  {h.cert_expires_in_s:.0f}s",
            "",
            "ANOMALY DETECTION",
            "-" * 50,
            f"  Score:            {h.anomaly_score:.4f}",
            f"  Raw Error:        {state.last_score:.4f}",
            f"  Threshold:        {state.threshold:.4f}",
            f"  EMA Mean:         {state.ema_mean:.4f}",
            f"  Detections:       {state.anomaly_count}",
            f"  Samples:          {state.sample_count}",
            "",
            "TRUST & SECURITY",
            "-" * 50,
            f"  Trust Level:      {h.trust_level}",
            f"  Peers Reachable:  {h.peers_reachable}",
            f"  CRL Entries:      {len(self._node.crl.get_all_revoked())}",
            "",
            "FEDERATION",
            "-" * 50,
            f"  Rounds:           {h.federation_rounds}",
            f"  Telemetry Ticks:  {h.telemetry_ticks}",
            "",
        ]
        return "\n".join(lines)


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics/health/status endpoints."""

    collector: Optional[MetricsCollector] = None

    def do_GET(self):
        if self.path == "/metrics":
            body = self.collector.collect_prometheus().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/health":
            data = self.collector.collect_health_json()
            body = json.dumps(data, indent=2).encode()
            status = 200 if data.get("online") else 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/status":
            body = self.collector.collect_status_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logs to avoid noise."""
        pass


def start_metrics_server(
    node: "SatelliteNode",
    port: int = METRICS_PORT,
) -> Tuple[HTTPServer, MetricsCollector]:
    """
    Start the metrics HTTP server in a daemon thread.

    Returns the HTTPServer and MetricsCollector so the caller can
    stop the server on shutdown.
    """
    collector = MetricsCollector(node)

    handler = type("Handler", (_MetricsHandler,), {"collector": collector})
    server = HTTPServer(("0.0.0.0", port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info("Metrics server started on http://0.0.0.0:%d", port)
    logger.info("  GET /metrics  — Prometheus scrape target")
    logger.info("  GET /health   — JSON health check")
    logger.info("  GET /status   — Human-readable status")

    return server, collector
