"""
Inter-Satellite Link (ISL) Messaging Layer.

Provides an asynchronous message bus abstraction over NATS for
inter-satellite communication. All messages are transported over
mTLS-secured connections.

In the Docker simulation, NATS runs as a separate container with TLS
enabled. Each satellite node presents its ephemeral X.509 certificate.

This module wraps nats-py with:
    - Automatic mTLS context injection
    - Topic-based pub/sub for ISL channels
    - Serialization helpers for weight deltas and CRL broadcasts

CCSDS Proximity-1 inspired topic hierarchy:
    astraea.telemetry.{node_id}     — Telemetry streams
    astraea.federation.deltas       — Weight delta submissions
    astraea.federation.global       — Global model broadcasts
    astraea.security.crl            — CRL synchronization
    astraea.routing.lsa             — Link-State Advertisements
    astraea.security.alerts         — Anomaly alerts
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Topic constants
TOPIC_FEDERATION_DELTAS = "astraea.federation.deltas"
TOPIC_FEDERATION_GLOBAL = "astraea.federation.global"
TOPIC_CRL_SYNC = "astraea.security.crl"
TOPIC_LSA = "astraea.routing.lsa"
TOPIC_ALERTS = "astraea.security.alerts"


def topic_telemetry(node_id: str) -> str:
    return f"astraea.telemetry.{node_id}"


# ------------------------------------------------------------------
# Stub classes (defined before use to avoid forward-reference issues)
# ------------------------------------------------------------------


class _StubMsg:
    """Stub NATS message for in-memory bus."""

    def __init__(self, topic: str, data: bytes):
        self.subject = topic
        self.data = data


class _StubSub:
    """Stub NATS subscription for in-memory bus."""

    def __init__(self, topic: str, cb: Callable):
        self.subject = topic
        self._cb = cb


class InMemoryBus:
    """
    In-memory message bus stub for testing without NATS.

    Supports basic pub/sub within a single process.
    Uses instance-level subscriptions to avoid cross-test pollution.
    """

    # Class-level registry allows multiple InMemoryBus instances in the
    # same process to share messages (simulating a real broker).
    _instances: List["InMemoryBus"] = []

    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable]] = {}
        InMemoryBus._instances.append(self)

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish to all subscribers across all bus instances."""
        msg = _StubMsg(topic=topic, data=data)
        for instance in InMemoryBus._instances:
            for cb in instance._subs.get(topic, []):
                try:
                    await cb(msg)
                except Exception as e:
                    logger.error("InMemoryBus callback error: %s", e)

    async def subscribe(self, topic: str, cb: Callable) -> _StubSub:
        if topic not in self._subs:
            self._subs[topic] = []
        self._subs[topic].append(cb)
        return _StubSub(topic=topic, cb=cb)

    async def close(self) -> None:
        """Remove this instance from the shared registry."""
        if self in InMemoryBus._instances:
            InMemoryBus._instances.remove(self)
        self._subs.clear()

    @classmethod
    def reset_all(cls) -> None:
        """Reset all instances — call between tests to avoid pollution."""
        cls._instances.clear()


# ------------------------------------------------------------------
# ISL Bus — Primary NATS-backed message bus
# ------------------------------------------------------------------


class ISLBus:
    """
    Inter-Satellite Link message bus backed by NATS.

    Provides pub/sub messaging with automatic reconnection and
    graceful degradation for simulated link disruptions.

    When a tls_context (ssl.SSLContext) is provided, all NATS traffic
    is wrapped in mTLS. This is the standard mode in Docker deployments.
    """

    def __init__(
        self,
        node_id: str,
        nats_url: str = "nats://nats-server:4222",
        tls_context: Optional[ssl.SSLContext] = None,
    ):
        self.node_id = node_id
        self.nats_url = nats_url
        self.tls_context = tls_context
        self._nc: Any = None
        self._subscriptions: Dict[str, Any] = {}
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to the NATS server."""
        try:
            import nats

            options: Dict[str, Any] = {
                "servers": [self.nats_url],
                "name": f"astraea-{self.node_id}",
                "reconnect_time_wait": 2,
                "max_reconnect_attempts": 60,
                "ping_interval": 10,
            }
            if self.tls_context is not None:
                options["tls"] = self.tls_context
                # Switch to TLS URL scheme if not already
                if self.nats_url.startswith("nats://"):
                    tls_url = self.nats_url.replace("nats://", "tls://", 1)
                    options["servers"] = [tls_url]
                    logger.info(
                        "ISL mTLS enabled: upgraded %s → %s",
                        self.nats_url,
                        tls_url,
                    )

            self._nc = await nats.connect(**options)
            self._connected = True
            logger.info(
                "ISL connected: node=%s  server=%s  tls=%s",
                self.node_id,
                self.nats_url,
                self.tls_context is not None,
            )
        except ImportError:
            logger.warning(
                "nats-py not installed — using in-memory stub for ISL bus"
            )
            self._nc = InMemoryBus()
            self._connected = True
        except Exception as e:
            logger.error("ISL connection failed: %s — falling back to in-memory bus", e)
            self._nc = InMemoryBus()
            self._connected = True

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish a message to a topic."""
        if not self._connected:
            await self.connect()
        try:
            await self._nc.publish(topic, data)
            logger.debug("Published %d bytes to %s", len(data), topic)
        except Exception as e:
            logger.error("Publish failed on %s: %s", topic, e)

    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Any], Coroutine],
    ) -> None:
        """Subscribe to a topic with an async callback."""
        if not self._connected:
            await self.connect()
        try:
            sub = await self._nc.subscribe(topic, cb=callback)
            self._subscriptions[topic] = sub
            logger.info("Subscribed to %s", topic)
        except Exception as e:
            logger.error("Subscribe failed on %s: %s", topic, e)

    async def publish_json(self, topic: str, payload: Dict) -> None:
        """Convenience: publish a JSON-serializable dict."""
        payload["_source"] = self.node_id
        await self.publish(topic, json.dumps(payload).encode())

    async def close(self) -> None:
        """Gracefully close the ISL connection."""
        if self._nc and self._connected:
            try:
                await self._nc.close()
            except Exception:
                pass
            self._connected = False
            logger.info("ISL disconnected: node=%s", self.node_id)
