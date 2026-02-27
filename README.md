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

Each node runs:
- **VAE Inference Engine** — Variational Autoencoder for telemetry anomaly detection
- **mTLS Identity** — Ephemeral ECDSA P-384 certs with SPIFFE IDs (300s TTL, auto-rotated)
- **Policy Decision Point** — Zero-Trust evaluation, certificate revocation
- **Federated Trainer** — Local VAE training, weight delta exchange (never raw data)
- **Link-State Router** — Dijkstra with trust-weighted edges
- **Prometheus Metrics** — `/metrics`, `/health`, `/status` on port 9090

---

## Quick Start

### Option 1: Just Run It (No Docker, 30 seconds)

```bash
# Install dependencies
pip install torch cryptography numpy pytest

# Run the test suite (29 tests)
make test
# or: python -m pytest tests/test_integration.py -v

# Run a red-team attack simulation
make red-team-polarized
# or: python -m scripts.red_team --attack polarized
```

That's it. This runs the full pipeline in-process: VAE training, anomaly detection,
PDP revocation, CRL update, and routing isolation. No Docker required.

### Option 2: Full Docker Constellation (5 nodes + NATS + traffic shaping)

```bash
# One command does everything: generates CA certs, builds images, starts containers
make docker-up

# Open Grafana dashboard in your browser
#   → http://localhost:3000  (login: admin / astraea)
#   The Astraea-1 Constellation dashboard loads automatically.

# Watch the live terminal dashboard (alternative to Grafana)
make dashboard

# Monitor raw logs
make docker-logs

# Run red-team attack inside a container
make docker-red-team

# Tear down
make docker-down
```

Or without Make:

```bash
# Step 1: Generate shared CA + per-node certificates (REQUIRED before Docker)
python -m scripts.bootstrap_ca

# Step 2: Build and launch the constellation
docker compose up --build -d

# Step 3: Monitor
open http://localhost:3000           # Grafana dashboard (admin / astraea)
docker compose logs -f
curl http://localhost:8222/varz      # NATS monitoring
curl http://localhost:9091/targets   # Prometheus scrape targets
curl http://localhost:9090/health    # Node health (JSON)
curl http://localhost:9090/metrics   # Prometheus metrics
curl http://localhost:9090/status    # Human-readable status

# Step 4: Attack!
docker compose exec sat-03 python -m scripts.red_team --attack polarized

# Step 5: Tear down
docker compose down
```

### Option 3: Red-Team with JSON Export

```bash
# Run all three attacks and save results
make red-team

# Or individually with output:
python -m scripts.red_team --attack polarized --output results/polarized.json
python -m scripts.red_team --attack spacejack --output results/spacejack.json
python -m scripts.red_team --attack bitflip   --output results/bitflip.json

# Results are machine-readable JSON:
cat results/polarized.json
```

---

## What Happens During an Attack

```
Nominal Telemetry → VAE detects reconstruction error spike
                  → Dynamic threshold exceeded (EMA + 3σ)
                  → PDP sees anomaly_score > 0.8 for 2 consecutive ticks
                  → Certificate REVOKED, serial added to CRL
                  → CRL broadcast to all peers over mTLS/NATS
                  → Routing engine sets trust=0, Dijkstra reroutes around node
                  → All mTLS connections from revoked node rejected
                  → Node fully isolated from constellation mesh
```

Detection-to-isolation time: **< 5ms** (SLA target: < 2s)

## Attack Models

| Attack | Real-World Analog | Detection Difficulty |
|---|---|---|
| `polarized` | Compromised sensor firmware slowly drifting readings | Hard — mimics orbital variation |
| `spacejack` | Hostile ground station spoofing commands | Medium — large multi-channel deviation |
| `bitflip` | Single Event Upset from cosmic radiation (SEU) | Easy — sudden spikes in readings |

---

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
├── observability/
│   ├── __init__.py          # Prometheus metrics + HTTP health server
│   └── logging.py           # Structured JSON/text logging
└── node/
    └── satellite_node.py    # Core node runtime (all subsystems)

scripts/
├── bootstrap_ca.py          # Generate shared CA + node certs for Docker
├── red_team.py              # Attack simulation (Deliverable #5)
├── dashboard.py             # Live terminal dashboard for monitoring
└── entrypoint.sh            # Docker entrypoint with tc traffic shaping

configs/
└── constellation.json       # Constellation topology + security + ML params

monitoring/
├── prometheus/prometheus.yml           # Scrape config for all 5 nodes
└── grafana/
    ├── dashboards/astraea-constellation.json  # Pre-built Grafana dashboard
    └── provisioning/                          # Auto-config datasource + dashboard

satellite_node.py            # Entry point — Deliverable #3
federated_aggregator.py      # Entry point — Deliverable #4
docker-compose.yaml          # Constellation orchestration — Deliverable #2
docs/ARCHITECTURE.md         # UML diagrams — Deliverable #1
Makefile                     # Common operations (make help for full list)
```

## Available Make Commands

```bash
make help                # Show all commands
make install             # Install Python dependencies
make test                # Run 29-test integration suite
make lint                # Syntax check all Python files
make red-team            # Run all 3 attack simulations with JSON export
make red-team-polarized  # Run polarized attack only
make docker-up           # Bootstrap CA + build + launch constellation
make docker-down         # Tear down constellation
make docker-logs         # Follow all node logs
make dashboard           # Open live terminal dashboard (Docker)
make dashboard-local     # Open dashboard for local testing
make clean               # Remove generated artifacts
```

## Observability

### Grafana Dashboard (UI)

When running with Docker (`make docker-up`), a full Grafana dashboard is available
at **http://localhost:3000** (login: `admin` / `astraea`).

The pre-built dashboard includes 12 panels across 3 sections:

**Constellation Overview** — node online/offline status, certificate TTL countdown,
trust level indicators, total telemetry throughput, CRL revocation count

**Anomaly Detection** — real-time anomaly score per node (with threshold lines
at 0.4/0.6/0.8), VAE reconstruction error vs dynamic threshold

**Trust & Security** — trust score timeseries per node, reachable peers in
routing mesh (drops when nodes get isolated)

**Operations** — node uptime, cumulative anomaly detections (stacked bar),
telemetry ingest rate (samples/sec)

All panels auto-refresh every 5 seconds.

### Per-Node HTTP Endpoints

Each satellite also exposes three endpoints on port 9090:

| Endpoint | Format | Use |
|---|---|---|
| `GET /metrics` | Prometheus text | Scrape target for Prometheus/Grafana |
| `GET /health` | JSON | Liveness/readiness probes (Docker, K8s) |
| `GET /status` | Plain text | Human-readable node status |

**Prometheus metrics exposed:**

- `astraea_anomaly_score` — Current normalized anomaly score [0-1]
- `astraea_anomaly_threshold` — Dynamic detection threshold
- `astraea_trust_score` — Node trust score [0-1]
- `astraea_trust_level` — Trust level (3=trusted, 0=revoked)
- `astraea_cert_ttl_remaining_seconds` — Time until cert expiry
- `astraea_telemetry_samples_total` — Total samples processed
- `astraea_anomaly_detections_total` — Total anomaly detections
- `astraea_crl_revoked_count` — Certificates in local CRL
- `astraea_routing_reachable_peers` — Reachable peer count
- `astraea_peer_trust_score{peer="sat-XX"}` — Per-peer trust

### Structured JSON Logging

Set `ASTRAEA_LOG_FORMAT=json` to switch all log output to single-line JSON
for ingestion by log aggregators (ELK, Loki, CloudWatch):

```bash
ASTRAEA_LOG_FORMAT=json ASTRAEA_LOG_LEVEL=DEBUG python -m astraea.node
```

## System Design Details

### Anomaly Detection Pipeline

1. **Telemetry Ingestion** — 3-channel vectors: `[Voltage, RW_RPM, SNR]` at 10 Hz
2. **VAE Reconstruction** — Forward pass computes `x_hat`, reconstruction error `L = ||x - x_hat||^2`
3. **Dynamic Threshold** — EMA-based: `tau = mu_ema + 3*sigma_ema` (only updated on nominal samples)
4. **Sigmoid Normalization** — Maps raw error to `[0, 1]` anomaly score
5. **Policy Evaluation** — Score > 0.8 for 2 consecutive ticks -> REVOKE

### Zero-Trust Mesh Isolation

When the PDP revokes a node:
1. Certificate serial added to CRL (thread-safe, broadcast to all peers)
2. Routing engine sets `trust_score = 0.0` -> all edges to node become infinite cost
3. Dijkstra recomputation bypasses the node via multi-hop alternate paths
4. mTLS connections from the revoked node are rejected by all peers

### Federated Learning

- **Algorithm**: FedAvg with L2-norm gradient clipping (clip_norm=10.0)
- **Privacy**: Only weight deltas (delta_w = w_local - w_global) are shared
- **Security**: Revoked nodes are excluded from aggregation rounds
- **Transport**: Serialized PyTorch state dicts over mTLS/NATS

### Certificate Lifecycle

- **Curve**: ECDSA P-384 (NIST Suite B compliant)
- **TTL**: 300 seconds (aggressive rotation for zero-trust)
- **Rotation**: Automatic at 80% TTL (60s before expiry)
- **SPIFFE ID**: `spiffe://astraea-1.mesh/satellite/{node_id}`
- **mTLS**: TLS 1.3 with mutual certificate verification

### Traffic Shaping (Docker)

Each container uses `tc` (Traffic Control) with HTB + netem:

| Node | Latency | Jitter | Loss | Bandwidth |
|---|---|---|---|---|
| SAT-01 | 200ms | 30ms | 0% | 2mbit |
| SAT-02 | 350ms | 50ms | 1% | 1mbit |
| SAT-03 | 400ms | 80ms | 2% | 1mbit |
| SAT-04 | 600ms | 100ms | 3% | 512kbit |
| SAT-05 | 800ms | 120ms | 5% | 512kbit |

## Configuration

All parameters are configurable via environment variables OR `configs/constellation.json`.
Environment variables take precedence.

| Variable | Default | Description |
|---|---|---|
| `ASTRAEA_NODE_ID` | `sat-01` | Node identifier |
| `ASTRAEA_NATS_URL` | `nats://nats-server:4222` | NATS broker URL |
| `ASTRAEA_PEERS` | *(from config)* | Comma-separated peer node IDs |
| `ASTRAEA_IS_AGGREGATOR` | `false` | Enable federation aggregator role |
| `ASTRAEA_ENABLE_MTLS` | `true` | Enable mTLS on NATS connections |
| `ASTRAEA_TELEMETRY_HZ` | `10` | Telemetry ingestion rate |
| `ASTRAEA_FEDERATION_INTERVAL` | `30` | Seconds between federation rounds |
| `ASTRAEA_METRICS_PORT` | `9090` | Prometheus metrics HTTP port |
| `ASTRAEA_LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `ASTRAEA_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

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

## Troubleshooting

**Tests fail with `ModuleNotFoundError: No module named 'torch'`**
```bash
pip install torch cryptography numpy pytest
```

**Docker fails with "certificate not found"**
```bash
# You need to run the CA bootstrap BEFORE docker compose up:
python -m scripts.bootstrap_ca
docker compose up --build -d
# Or just: make docker-up  (does both automatically)
```

**NATS connection refused**
```bash
# Check NATS is healthy:
docker compose ps
curl http://localhost:8222/varz
```

**Want to change the constellation topology?**
Edit `configs/constellation.json` — it defines all node IDs, peers, orbital
parameters, security settings, and ML hyperparameters.

## CI/CD

GitHub Actions runs on every push and pull request:
- Syntax check across all Python files
- Full 29-test integration suite (Python 3.10, 3.11, 3.12)
- All three red-team attack simulations
- Docker build verification

## License

MIT
