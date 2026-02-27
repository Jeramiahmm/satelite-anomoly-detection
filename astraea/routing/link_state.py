"""
Trust-Aware Link-State Routing for the Astraea-1 Constellation.

Implements a Dijkstra-based shortest-path algorithm where edge weights
are inversely proportional to the trust score of the destination node.
When a node's trust score drops to zero (revoked), all edges to that
node become infinite — effectively severing it from the mesh.

This is the SD-WAN layer: routing adapts in real-time to the security
posture of the constellation, bypassing compromised nodes via multi-hop
paths through trusted peers.

Edge weight formula:
    w(i, j) = base_latency(i, j) / trust_score(j)
    If trust_score(j) == 0 → w(i, j) = ∞

Reference: NIST SP 800-207 §4.3 — "Network-based: micro-segmentation"
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

INF = float("inf")


@dataclass
class LinkInfo:
    """Metadata about a link to a neighboring node."""

    peer_id: str
    base_latency_ms: float  # Simulated propagation delay
    trust_score: float = 1.0  # [0.0, 1.0]
    last_lsa_time: float = 0.0  # Last Link-State Advertisement
    cert_serial: Optional[int] = None


@dataclass
class RoutingEntry:
    """A single entry in the forwarding table."""

    destination: str
    next_hop: str
    cost: float
    path: List[str]


class LinkStateRouter:
    """
    Trust-aware link-state router for an individual satellite node.

    Maintains a local view of the constellation topology and recomputes
    shortest paths whenever trust scores change or links are updated.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # Adjacency: node_id → {peer_id → LinkInfo}
        self._topology: Dict[str, Dict[str, LinkInfo]] = {node_id: {}}
        self._forwarding_table: Dict[str, RoutingEntry] = {}
        self._lock = threading.Lock()
        self._revision = 0

    def add_link(
        self,
        from_node: str,
        to_node: str,
        base_latency_ms: float,
        trust_score: float = 1.0,
        cert_serial: Optional[int] = None,
    ) -> None:
        """Add or update a link in the topology graph."""
        with self._lock:
            if from_node not in self._topology:
                self._topology[from_node] = {}
            self._topology[from_node][to_node] = LinkInfo(
                peer_id=to_node,
                base_latency_ms=base_latency_ms,
                trust_score=trust_score,
                last_lsa_time=time.time(),
                cert_serial=cert_serial,
            )
            # Ensure the destination also exists in topology
            if to_node not in self._topology:
                self._topology[to_node] = {}

    def update_trust_score(self, node_id: str, trust_score: float) -> None:
        """
        Update the trust score for a node, affecting all edges TO that node.

        If trust_score == 0.0, the node is effectively isolated.
        Triggers a routing recomputation.
        """
        with self._lock:
            changed = False
            for from_node, neighbors in self._topology.items():
                if node_id in neighbors:
                    old_score = neighbors[node_id].trust_score
                    if old_score != trust_score:
                        neighbors[node_id].trust_score = trust_score
                        changed = True
            if changed:
                self._revision += 1
                logger.info(
                    "Trust score updated: node=%s  score=%.2f  revision=%d",
                    node_id,
                    trust_score,
                    self._revision,
                )
        if changed:
            self.recompute_routes()

    def isolate_node(self, node_id: str) -> None:
        """Set trust to 0.0 — sever all routes through this node."""
        logger.warning("ISOLATING node %s from routing mesh", node_id)
        self.update_trust_score(node_id, 0.0)

    def recompute_routes(self) -> Dict[str, RoutingEntry]:
        """
        Run Dijkstra from this node using trust-weighted edge costs.

        Edge cost: base_latency / trust_score
        trust_score == 0 → cost = ∞ (unreachable)
        """
        with self._lock:
            graph = self._topology
            source = self.node_id

            # Dijkstra
            dist: Dict[str, float] = {source: 0.0}
            prev: Dict[str, Optional[str]] = {source: None}
            visited: Set[str] = set()
            heap: List[Tuple[float, str]] = [(0.0, source)]

            all_nodes = set(graph.keys())
            for neighbors in graph.values():
                for peer_id in neighbors:
                    all_nodes.add(peer_id)

            for node in all_nodes:
                if node != source:
                    dist[node] = INF
                    prev[node] = None

            while heap:
                d, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)

                for peer_id, link in graph.get(u, {}).items():
                    if link.trust_score <= 0.0:
                        continue  # Node is isolated
                    edge_cost = link.base_latency_ms / link.trust_score
                    alt = d + edge_cost
                    if alt < dist.get(peer_id, INF):
                        dist[peer_id] = alt
                        prev[peer_id] = u
                        heapq.heappush(heap, (alt, peer_id))

            # Build forwarding table
            new_table: Dict[str, RoutingEntry] = {}
            for dest in all_nodes:
                if dest == source:
                    continue
                if dist.get(dest, INF) == INF:
                    continue  # Unreachable

                # Trace path back to find next hop
                path = []
                current = dest
                while current is not None:
                    path.append(current)
                    current = prev.get(current)
                path.reverse()

                if len(path) >= 2:
                    next_hop = path[1]
                    new_table[dest] = RoutingEntry(
                        destination=dest,
                        next_hop=next_hop,
                        cost=dist[dest],
                        path=path,
                    )

            self._forwarding_table = new_table
            self._revision += 1

        logger.debug(
            "Routes recomputed: %d reachable destinations  revision=%d",
            len(new_table),
            self._revision,
        )
        return dict(new_table)

    def get_next_hop(self, destination: str) -> Optional[str]:
        """Look up the next hop for a given destination."""
        with self._lock:
            entry = self._forwarding_table.get(destination)
            return entry.next_hop if entry else None

    def get_route(self, destination: str) -> Optional[RoutingEntry]:
        """Get the full routing entry for a destination."""
        with self._lock:
            return self._forwarding_table.get(destination)

    def get_reachable_nodes(self) -> Set[str]:
        """Return set of all reachable node IDs."""
        with self._lock:
            return set(self._forwarding_table.keys())

    def get_topology_summary(self) -> Dict:
        """Return a JSON-serializable topology summary for debugging."""
        with self._lock:
            summary = {
                "node_id": self.node_id,
                "revision": self._revision,
                "links": {},
                "forwarding_table": {},
            }
            for from_node, neighbors in self._topology.items():
                summary["links"][from_node] = {
                    peer: {
                        "latency_ms": info.base_latency_ms,
                        "trust": info.trust_score,
                    }
                    for peer, info in neighbors.items()
                }
            for dest, entry in self._forwarding_table.items():
                summary["forwarding_table"][dest] = {
                    "next_hop": entry.next_hop,
                    "cost": entry.cost,
                    "path": entry.path,
                }
            return summary

    def generate_lsa(self) -> Dict:
        """
        Generate a Link-State Advertisement for broadcast to peers.

        Contains this node's direct neighbors and their trust scores.
        """
        with self._lock:
            my_links = self._topology.get(self.node_id, {})
            return {
                "type": "lsa",
                "origin": self.node_id,
                "revision": self._revision,
                "timestamp": time.time(),
                "neighbors": {
                    peer: {
                        "latency_ms": info.base_latency_ms,
                        "trust": info.trust_score,
                    }
                    for peer, info in my_links.items()
                },
            }

    def apply_lsa(self, lsa: Dict) -> None:
        """Apply a Link-State Advertisement received from a peer."""
        origin = lsa["origin"]
        for peer_id, info in lsa.get("neighbors", {}).items():
            self.add_link(
                from_node=origin,
                to_node=peer_id,
                base_latency_ms=info["latency_ms"],
                trust_score=info["trust"],
            )
        self.recompute_routes()
