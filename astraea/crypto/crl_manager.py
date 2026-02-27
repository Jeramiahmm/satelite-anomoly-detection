"""
Certificate Revocation List (CRL) Manager.

Provides a distributed CRL cache that each satellite node maintains locally.
CRL updates are propagated over the ISL message bus, ensuring that revoked
nodes are isolated from the mesh within the detection-to-isolation SLA (< 2s).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CRLCache:
    """
    Local CRL cache maintained by each satellite node.

    Thread-safe. Supports subscription for revocation events so that the
    routing layer can immediately recompute paths.
    """

    _revoked: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _revision: int = field(default=0)
    _listeners: list = field(default_factory=list)

    def add_revocation(self, serial_number: int, source: str = "local") -> bool:
        """
        Add a serial number to the local CRL.

        Returns True if this was a new revocation (not already cached).
        """
        with self._lock:
            if serial_number in self._revoked:
                return False
            self._revoked.add(serial_number)
            self._revision += 1
            logger.warning(
                "CRL updated: revoked serial=%d  source=%s  revision=%d",
                serial_number,
                source,
                self._revision,
            )
        # Fire listeners outside the lock to avoid deadlocks
        for listener in self._listeners:
            try:
                listener(serial_number)
            except Exception:
                logger.exception("CRL listener error")
        return True

    def is_revoked(self, serial_number: int) -> bool:
        with self._lock:
            return serial_number in self._revoked

    def get_all_revoked(self) -> frozenset:
        with self._lock:
            return frozenset(self._revoked)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def on_revocation(self, callback: Callable[[int], None]) -> None:
        """Register a callback invoked on each new revocation event."""
        self._listeners.append(callback)

    def merge_from_broadcast(self, revoked_serials: set, source: str = "peer") -> int:
        """
        Merge a set of revoked serials received from a peer broadcast.

        Returns the number of newly added revocations.
        """
        added = 0
        for serial in revoked_serials:
            if self.add_revocation(serial, source=source):
                added += 1
        return added

    def serialize(self) -> bytes:
        """Serialize the CRL cache for ISL broadcast."""
        with self._lock:
            payload = {
                "type": "crl_sync",
                "revision": self._revision,
                "revoked_serials": list(self._revoked),
                "timestamp": time.time(),
            }
        return json.dumps(payload).encode()

    @staticmethod
    def deserialize_broadcast(data: bytes) -> set:
        """Deserialize a CRL sync payload from a peer."""
        payload = json.loads(data.decode())
        return set(payload.get("revoked_serials", []))
