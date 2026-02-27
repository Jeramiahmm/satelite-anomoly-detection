"""
Constellation CA Bootstrap Script.

Generates a shared Certificate Authority and per-node certificates
for use in Docker deployments where all nodes must trust the same CA.

This runs ONCE before `docker compose up` and writes all identity
materials to a shared volume mount (./certs/).

Usage:
    python -m scripts.bootstrap_ca
    docker compose up --build
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astraea.crypto.identity import (
    generate_ca,
    issue_node_certificate,
    persist_identity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap_ca")

NODE_IDS = ["sat-01", "sat-02", "sat-03", "sat-04", "sat-05"]
CERT_DIR = Path(__file__).resolve().parent.parent / "certs"


def bootstrap() -> None:
    """Generate shared CA and per-node certificates."""
    logger.info("Generating constellation CA...")
    ca = generate_ca()

    # Write CA cert and key to shared directory
    CERT_DIR.mkdir(parents=True, exist_ok=True)

    from cryptography.hazmat.primitives import serialization

    ca_cert_path = CERT_DIR / "ca.crt"
    ca_key_path = CERT_DIR / "ca.key"

    ca_cert_path.write_bytes(
        ca.certificate.public_bytes(serialization.Encoding.PEM)
    )
    ca_key_path.write_bytes(
        ca.private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    import os
    os.chmod(ca_key_path, 0o600)

    logger.info("CA written to %s", CERT_DIR)

    # Issue per-node certificates
    for node_id in NODE_IDS:
        identity = issue_node_certificate(ca, node_id)
        node_dir = CERT_DIR / node_id
        persist_identity(identity, node_dir)
        logger.info("Issued cert for %s → %s", node_id, node_dir)

    # Also generate a NATS server certificate
    nats_identity = issue_node_certificate(
        ca, "nats-server", san_ip="172.28.0.10", ttl_seconds=86400
    )
    nats_dir = CERT_DIR / "nats-server"
    persist_identity(nats_identity, nats_dir)
    logger.info("Issued NATS server cert → %s", nats_dir)

    logger.info("Bootstrap complete. Run: docker compose up --build")


if __name__ == "__main__":
    bootstrap()
