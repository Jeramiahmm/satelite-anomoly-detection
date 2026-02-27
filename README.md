# Astraea-1 Protocol

**Decentralized Zero-Trust Intrusion Detection System for LEO Satellite Swarms**

A production-grade simulation of federated anomaly detection with mutual TLS,
SPIFFE-inspired identity, and trust-aware routing for software-defined satellite
constellations.

## Reference Standards

| Standard | Application |
|---|---|
| **NIST SP 800-207** | Zero Trust Architecture — PDP/PEP/PIP model |
| **CCSDS 350.0-G-3** | Space Data Link Security Protocol |
| **SPIFFE/SPIRE** | Workload identity — ephemeral X.509 certificates |
| **RFC 5280** | X.509 PKI Certificate and CRL Profile |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASTRAEA-1 CONSTELLATION MESH                     │
│                                                                     │
│  ┌──────────┐   mTLS/NATS    ┌──────────┐   mTLS/NATS             │
│  │  SAT-01  │◄──────────────►│  SAT-02  │◄──────────┐             │
│  │  (Node)  │                │  (Node)  │           │             │
│  └────┬─────┘                └────┬─────┘           │             │
│       │                           │                  │             │
│  ┌────▼─────┐                ┌────▼─────┐    ┌──────▼───┐        │
│  │  SAT-03  │◄──────────────►│  SAT-04  │◄──►│  SAT-05  │        │
│  │  (Node)  │   mTLS/NATS   │  (Node)  │    │ (Aggreg) │        │
│  └──────────┘                └──────────┘    └──────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

Each node integrates:
- **VAE Inference Engine** — Variational Autoencoder for telemetry anomaly detection
- **mTLS Identity** — Ephemeral ECDSA P-384 certs with SPIFFE IDs (300s TTL)
- **Policy Decision Point** — Zero-Trust evaluation, certificate revocation
- **Federated Trainer** — Local VAE training, weight delta exchange (never raw data)
- **Link-State Router** — Dijkstra with trust-weighted edges

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for constellation simulation)

### Run Tests (No Docker Required)

```bash
pip install torch cryptography numpy pytest
python -m pytest tests/test_integration.py -v
```

### Run Red-Team Attack Simulation (No Docker Required)

```bash
# Polarized Telemetry Injection (hardest to detect)
python -m scripts.red_team --attack polarized

# Space-Jacking attack
python -m scripts.red_team --attack spacejack

# Bit-Flip attack (simulating SEU)
python -m scripts.red_team --attack bitflip

# Custom parameters
python -m scripts.red_team --target sat-03 --attack polarized --intensity 0.8 --nodes 5 --sla 2.0
```

### Run Full Docker Constellation

```bash
# Build and launch 5-node swarm with NATS broker
docker compose up --build -d

# Monitor logs
docker compose logs -f

# Run red-team attack inside a container
docker compose exec sat-03 python -m scripts.red_team

# Inspect NATS monitoring
curl http://localhost:8222/varz

# Tear down
docker compose down
```

## Project Structure

```
astraea/
├── crypto/
│   ├── identity.py          # SPIFFE/X.509 identity, CA, mTLS contexts
│   └── crl_manager.py       # Distributed Certificate Revocation List
├── ml/
│   ├── vae.py               # Variational Autoencoder (PyTorch)
│   ├── anomaly_detector.py  # Dynamic threshold engine (EMA-based)
│   ├── telemetry.py         # Synthetic telemetry generator + attack modes
│   └── trainer.py           # Local VAE trainer + weight delta computation
├── federation/
│   └── aggregator.py        # FedAvg coordinator with gradient clipping
├── routing/
│   └── link_state.py        # Trust-aware Dijkstra routing
├── messaging/
│   └── isl.py               # NATS-backed Inter-Satellite Link bus
├── policy/
│   └── pdp.py               # Policy Decision Point (NIST 800-207)
└── node/
    └── satellite_node.py    # Core node runtime (all subsystems)

satellite_node.py            # Entry point — Deliverable #3
federated_aggregator.py      # Entry point — Deliverable #4
scripts/red_team.py          # Attack simulation — Deliverable #5
docker-compose.yaml          # Constellation orchestration — Deliverable #2
docs/ARCHITECTURE.md         # UML diagrams — Deliverable #1
```

## Key Deliverables

| # | Deliverable | File | Description |
|---|---|---|---|
| 1 | UML Diagram | `docs/ARCHITECTURE.md` | Mermaid sequence diagram: telemetry → detection → isolation |
| 2 | Docker Compose | `docker-compose.yaml` | 5-node swarm + NATS + traffic shaping (tc) |
| 3 | Satellite Node | `satellite_node.py` | VAE engine + mTLS server + routing table |
| 4 | Fed. Aggregator | `federated_aggregator.py` | Secure FedAvg with gradient clipping |
| 5 | Red-Team Script | `scripts/red_team.py` | Polarized telemetry injection attack |

## System Design Details

### Anomaly Detection Pipeline

1. **Telemetry Ingestion** — 3-channel vectors: `[Voltage, RW_RPM, SNR]` at 10 Hz
2. **VAE Reconstruction** — Forward pass computes `x_hat`, reconstruction error `L = ||x - x_hat||^2`
3. **Dynamic Threshold** — EMA-based: `tau = mu_ema + 3*sigma_ema` (only updated on nominal samples)
4. **Sigmoid Normalization** — Maps raw error to `[0, 1]` anomaly score
5. **Policy Evaluation** — Score > 0.8 for 2 consecutive ticks → REVOKE

### Zero-Trust Mesh Isolation

When the PDP revokes a node:
1. Certificate serial added to CRL (thread-safe, broadcast to all peers)
2. Routing engine sets `trust_score = 0.0` → all edges to node become infinite
3. Dijkstra recomputation bypasses the node via multi-hop alternate paths
4. mTLS connections from the revoked node are rejected by all peers

### Federated Learning

- **Algorithm**: FedAvg with L2-norm gradient clipping
- **Privacy**: Only weight deltas (delta_w = w_local - w_global) are shared
- **Security**: Revoked nodes are excluded from aggregation rounds
- **Transport**: Serialized PyTorch state dicts over mTLS/NATS

### Traffic Shaping (Docker)

Each container uses `tc` (Traffic Control) with HTB + netem:
- **Latency**: 200ms-800ms (variable per orbital position)
- **Jitter**: 30ms-120ms (Gaussian distribution)
- **Loss**: 0%-5% (Grey-Hole attack simulation on SAT-05)
- **Bandwidth**: 512kbit-2mbit (ISL capacity constraints)

## Test Results

```
29 passed — Full integration test suite
├── TestCryptoIdentity (5 tests)     — X.509, SPIFFE, CRL
├── TestCRLCache (5 tests)           — Distributed revocation
├── TestVAE (4 tests)                — Model shape, loss, reconstruction
├── TestAnomalyDetector (2 tests)    — Nominal baseline, attack detection
├── TestFederation (4 tests)         — FedAvg, revocation, clipping
├── TestRouting (4 tests)            — Dijkstra, isolation, trust scoring
├── TestPolicyDecisionPoint (4 tests)— Trust levels, revocation triggers
└── TestEndToEndPipeline (1 test)    — Full D→I path < 2s SLA
```

Red-Team results (all PASS):

| Attack | Detection | D-to-I Time | SLA Met |
|---|---|---|---|
| Spacejack | 0.8ms | 1.8ms | < 2.0s |
| Polarized | 0.9ms | 2.2ms | < 2.0s |
| Bitflip | 1.2ms | 3.5ms | < 2.0s |
