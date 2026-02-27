"""
Satellite Node — Standalone Entry Point (Deliverable #3)
=========================================================

Top-level launcher for a single satellite node in the Astraea-1
constellation. Combines:
    - VAE Inference Engine (anomaly detection)
    - mTLS Server (SPIFFE-inspired identity)
    - Routing Table (trust-weighted link-state)
    - Federated Trainer (local training + weight delta exchange)
    - Policy Decision Point (zero-trust evaluation)

Configuration is loaded from environment variables (see NodeConfig.from_env).

Usage:
    # Standalone
    ASTRAEA_NODE_ID=sat-01 python satellite_node.py

    # Docker (handled by entrypoint.sh)
    docker compose up sat-01
"""

from astraea.node.satellite_node import main

if __name__ == "__main__":
    main()
