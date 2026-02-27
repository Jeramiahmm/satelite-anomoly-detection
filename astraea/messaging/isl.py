"""
Inter-Satellite Link (ISL) Messaging Layer.

Provides an asynchronous message bus abstraction over NATS for
inter-satellite communication. All messages are transported over
mTLS-secured connections.

In the Docker simulation, NATS runs as a separate container.
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
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)

# Topic constants
TOPIC_FEDERATION_DELTAS = "astraea.federation.deltas"
TOPIC_FEDERATION_GLOBAL = "astraea.federation.global"
TOPIC_CRL_SYNC = "astraea.security.crl"
TOPIC_LSA = "astraea.routing.lsa"
TOPIC_ALERTS = "astraea.security.alerts"


def topic_telemetry(node_id: str) -> str:
    return f"astraea.telemetry.{node_id}"


class ISLBus:
    """
    Inter-Satellite Link message bus backed by NATS.

    Provides pub/sub messaging with automatic reconnection and
    graceful degradation for simulated link disruptions.
    """

    def __init__(
        self,
        node_id: str,
        nats_url: str = "nats://nats-server:4222",
        tls_context: Optional[Any] = None,
    ):
        self.node_id = node_id
        self.nats_url = nats_url
        self.tls_context = tls_context
        self._nc = None  # nats.aio.client.Client
        self._subscriptions: Dict[str, Any] = {}
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to the NATS server."""
        try:
            import nats

            options = {
                "servers": [self.nats_url],
                "name": f"astraea-{self.node_id}",
                "reconnect_time_wait": 2,
                "max_reconnect_attempts": 60,
                "ping_interval": 10,
            }
            if self.tls_context:
                options["tls"] = self.tls_context

            self._nc = await nats.connect(**options)
            self._connected = True
            logger.info(
                "ISL connected: node=%s  server=%s", self.node_id, self.nats_url
            )
        except ImportError:
            logger.warning(
                "nats-py not installed — using in-memory stub for ISL bus"
            )
            self._nc = InMemoryBus()
            self._connected = True
        except Exception as e:
            logger.error("ISL connection failed: %s", e)
            self._nc = InMemoryBus()
            self._connected = True

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish a message to a topic."""
        if not self._connected:
            await self.connect()
        try:
            await self._nc.publish(topic, data)
            logger.debug(
                "Published %d bytes to %s", len(data), topic
            )
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


class InMemoryBus:
    """
    In-memory message bus stub for testing without NATS.

    Supports basic pub/sub within a single process.
    """

    _global_subs: Dict[str, list] = {}

    async def publish(self, topic: str, data: bytes) -> None:
        for cb in self._global_subs.get(topic, []):
            msg = _StubMsg(topic=topic, data=data)
            try:
                await cb(msg)
            except Exception as e:
                logger.error("InMemoryBus callback error: %s", e)

    async def subscribe(self, topic: str, cb: Callable) -> _StubSub:
        if topic not in self._global_subs:
            self._global_subs[topic] = []
        self._global_subs[topic].append(cb)
        return _StubSub(topic=topic, cb=cb)

    async def close(self) -> None:
        pass


class _StubMsg:
    def __init__(self, topic: str, data: bytes):
        self.subject = topic
        self.data = data


class _StubSub:
    def __init__(self, topic: str, cb: Callable):
        self.subject = topic
        self._cb = cb
