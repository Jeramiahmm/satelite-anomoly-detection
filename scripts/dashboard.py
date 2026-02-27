"""
Live Terminal Dashboard for Astraea-1 Constellation.

Connects to the /health endpoint of each satellite node and displays
a real-time terminal dashboard showing anomaly scores, trust levels,
certificate TTL, and routing status.

Usage:
    # Monitor local Docker constellation:
    python -m scripts.dashboard

    # Monitor specific nodes:
    python -m scripts.dashboard --nodes sat-01,sat-02,sat-03

    # Custom refresh rate:
    python -m scripts.dashboard --interval 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NodeStatus:
    node_id: str
    online: bool = False
    uptime_s: float = 0.0
    cert_expires_in_s: float = 0.0
    anomaly_score: float = 0.0
    trust_level: str = "unknown"
    peers_reachable: int = 0
    federation_rounds: int = 0
    telemetry_ticks: int = 0
    error: str = ""


# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def _trust_color(level: str) -> str:
    return {
        "trusted": C.GREEN,
        "degraded": C.YELLOW,
        "untrusted": C.RED,
        "revoked": C.RED + C.BOLD,
    }.get(level, C.DIM)


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    if score > 0.8:
        color = C.RED
    elif score > 0.6:
        color = C.YELLOW
    else:
        color = C.GREEN
    bar = color + "|" * filled + C.DIM + "." * (width - filled) + C.RESET
    return bar


def _cert_color(remaining: float) -> str:
    if remaining <= 0:
        return C.RED + C.BOLD
    elif remaining < 60:
        return C.YELLOW
    return C.GREEN


def fetch_node_health(base_url: str) -> NodeStatus:
    """Fetch /health from a node's metrics server."""
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return NodeStatus(**{k: v for k, v in data.items() if k != "error"})
    except urllib.error.URLError as e:
        return NodeStatus(node_id="?", error=str(e.reason))
    except Exception as e:
        return NodeStatus(node_id="?", error=str(e))


def render_dashboard(statuses: Dict[str, NodeStatus], tick: int) -> str:
    """Render the terminal dashboard as a string."""
    lines = []
    now = time.strftime("%H:%M:%S")

    lines.append("")
    lines.append(f"{C.BOLD}{C.CYAN}  ASTRAEA-1 CONSTELLATION DASHBOARD{C.RESET}  [{now}]  tick={tick}")
    lines.append(f"{C.DIM}  {'=' * 72}{C.RESET}")
    lines.append("")

    # Header
    lines.append(
        f"  {C.BOLD}{'NODE':<10} {'STATUS':<9} {'TRUST':<11} "
        f"{'ANOMALY':<8} {'SCORE BAR':<24} {'CERT TTL':<10} "
        f"{'PEERS':<7} {'TICKS':<8}{C.RESET}"
    )
    lines.append(f"  {C.DIM}{'-' * 72}{C.RESET}")

    for node_id, s in sorted(statuses.items()):
        if s.error:
            lines.append(f"  {node_id:<10} {C.RED}OFFLINE{C.RESET}   {C.DIM}{s.error[:50]}{C.RESET}")
            continue

        # Status
        status_str = f"{C.GREEN}ONLINE{C.RESET}" if s.online else f"{C.RED}DOWN{C.RESET}"

        # Trust
        tc = _trust_color(s.trust_level)
        trust_str = f"{tc}{s.trust_level.upper():<9}{C.RESET}"

        # Anomaly score
        score_str = f"{s.anomaly_score:.4f}"
        if s.anomaly_score > 0.8:
            score_str = f"{C.RED}{score_str}{C.RESET}"
        elif s.anomaly_score > 0.4:
            score_str = f"{C.YELLOW}{score_str}{C.RESET}"

        # Score bar
        bar = _score_bar(s.anomaly_score)

        # Cert TTL
        cc = _cert_color(s.cert_expires_in_s)
        cert_str = f"{cc}{s.cert_expires_in_s:.0f}s{C.RESET}"

        # Peers
        peers_str = f"{s.peers_reachable}"

        # Ticks
        ticks_str = f"{s.telemetry_ticks}"

        lines.append(
            f"  {node_id:<10} {status_str:<18} {trust_str:<20} "
            f"{score_str:<17} {bar}  {cert_str:<19} {peers_str:<7} {ticks_str:<8}"
        )

    lines.append("")
    lines.append(f"  {C.DIM}{'=' * 72}{C.RESET}")

    # Legend
    lines.append(
        f"  {C.DIM}Anomaly: "
        f"{C.GREEN}|||||{C.DIM}=safe  "
        f"{C.YELLOW}|||||{C.DIM}=degraded  "
        f"{C.RED}|||||{C.DIM}=critical  "
        f"  Ctrl+C to exit{C.RESET}"
    )
    lines.append("")

    return "\n".join(lines)


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Astraea-1 Live Dashboard")
    parser.add_argument(
        "--nodes",
        default="sat-01,sat-02,sat-03,sat-04,sat-05",
        help="Comma-separated node IDs to monitor",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=9090,
        help="Metrics port (each node uses base_port offset by node index)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to connect to (default: localhost)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker container names as hostnames (e.g., astraea-sat-01)",
    )
    args = parser.parse_args()

    node_ids = [n.strip() for n in args.nodes.split(",") if n.strip()]

    # Build URL map
    urls: Dict[str, str] = {}
    for i, nid in enumerate(node_ids):
        if args.docker:
            host = f"astraea-{nid}"
            urls[nid] = f"http://{host}:{args.base_port}"
        else:
            # When running locally, offset port per node
            port = args.base_port + i
            urls[nid] = f"http://{args.host}:{port}"

    print(f"{C.CYAN}Astraea-1 Dashboard — monitoring {len(node_ids)} nodes{C.RESET}")
    print(f"{C.DIM}Press Ctrl+C to stop{C.RESET}\n")

    tick = 0
    try:
        while True:
            statuses: Dict[str, NodeStatus] = {}
            for nid, url in urls.items():
                s = fetch_node_health(url)
                if not s.node_id or s.node_id == "?":
                    s.node_id = nid
                statuses[nid] = s

            clear_screen()
            print(render_dashboard(statuses, tick))
            tick += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Dashboard stopped.{C.RESET}")


if __name__ == "__main__":
    main()
