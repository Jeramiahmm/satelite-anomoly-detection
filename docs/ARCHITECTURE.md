# Astraea-1 Protocol — Architecture Document

## Reference Standards
- **NIST SP 800-207**: Zero Trust Architecture
- **CCSDS 350.0-G-3**: Space Data Link Security Protocol
- **SPIFFE/SPIRE**: Secure Production Identity Framework for Everyone

## Unified System Lifecycle — UML Sequence Diagram

```mermaid
sequenceDiagram
    participant TG as Telemetry Generator
    participant SN as Satellite Node (VAE Engine)
    participant ISL as Inter-Satellite Link (NATS/mTLS)
    participant FA as Federated Aggregator
    participant PDP as Policy Decision Point
    participant RT as Routing Table (Link-State)
    participant CRL as Certificate Revocation List

    Note over TG,CRL: ── Phase 1: Identity Bootstrap (SPIFFE-Inspired) ──
    SN->>SN: Generate ephemeral ECDSA P-384 keypair
    SN->>PDP: CSR with SPIFFE ID (spiffe://astraea-1.mesh/satellite/{node_id})
    PDP->>SN: Signed X.509 cert (TTL: 300s, auto-rotate)
    SN->>ISL: Register with mTLS handshake (mutual authentication)

    Note over TG,CRL: ── Phase 2: Telemetry Ingestion & Local Inference ──
    TG->>SN: Raw telemetry vector [Voltage, RW_RPM, SNR] @ 10Hz
    SN->>SN: VAE forward pass → reconstruction x̂
    SN->>SN: Compute reconstruction error: L = ||x - x̂||²
    SN->>SN: Compare L against dynamic threshold τ (EMA-based)

    alt L < τ (Normal)
        SN->>SN: Update local trust score → 1.0
        SN->>RT: Advertise trust_score=1.0 via LSA
    else L ≥ τ (Anomaly Detected)
        SN->>PDP: ALERT {node_id, anomaly_score, timestamp}
        PDP->>PDP: Evaluate policy: score > 0.8 → REVOKE

        Note over PDP,CRL: ── Phase 3: Zero-Trust Mesh Isolation ──
        PDP->>CRL: Add serial_number to CRL
        PDP->>ISL: Broadcast REVOCATION_EVENT to all peers
        ISL->>SN: Peer validates CRL → reject connections from revoked node
        PDP->>RT: Set trust_score=0.0 for compromised node
    end

    Note over TG,CRL: ── Phase 4: Federated Learning (Secure Weight Exchange) ──
    SN->>SN: Compute weight delta: Δw = w_local - w_global
    SN->>ISL: Send Δw via mTLS side-channel (NEVER raw telemetry)
    ISL->>FA: Aggregate Δw from all trusted (CRL-clean) nodes
    FA->>FA: FedAvg: w_global_new = w_global + (1/n) Σ Δw_i
    FA->>ISL: Broadcast w_global_new to constellation
    ISL->>SN: Apply w_global_new → w_local

    Note over TG,CRL: ── Phase 5: Adaptive Routing ──
    RT->>RT: Recompute Dijkstra with edge weight = 1/trust_score
    RT->>SN: Update forwarding table (bypass untrusted nodes)
    SN->>ISL: Route traffic via next-best multi-hop path
```

## Component Interaction Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASTRAEA-1 CONSTELLATION MESH                     │
│                                                                     │
│  ┌──────────┐   mTLS/NATS    ┌──────────┐   mTLS/NATS             │
│  │  SAT-01  │◄──────────────►│  SAT-02  │◄──────────┐             │
│  │  (Node)  │                │  (Node)  │           │             │
│  └────┬─────┘                └────┬─────┘           │             │
│       │                           │                  │             │
│       │ mTLS/NATS                 │ mTLS/NATS       │             │
│       │                           │                  │             │
│  ┌────▼─────┐                ┌────▼─────┐    ┌──────▼───┐        │
│  │  SAT-03  │◄──────────────►│  SAT-04  │◄──►│  SAT-05  │        │
│  │  (Node)  │   mTLS/NATS   │  (Node)  │    │ (Aggreg) │        │
│  └──────────┘                └──────────┘    └──────────┘        │
│                                                                     │
│  Each Node Contains:                                                │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌────────┐ │           │
│  │  │   VAE   │  │  mTLS    │  │ Route │  │  PDP   │ │           │
│  │  │ Engine  │  │ Identity │  │ Table │  │ Engine │ │           │
│  │  └────┬────┘  └────┬─────┘  └───┬───┘  └───┬────┘ │           │
│  │       └─────────────┴────────────┴──────────┘      │           │
│  │                  Message Bus (NATS)                  │           │
│  └─────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Trust Score Computation

```
trust_score(node_i) = max(0, 1.0 - α * anomaly_score(node_i))

Where:
  α = sensitivity coefficient (default 1.25)
  anomaly_score = normalized VAE reconstruction error
  trust_score ∈ [0.0, 1.0]

Routing edge weight: w(i,j) = base_latency / trust_score(j)
  If trust_score(j) = 0 → w(i,j) = ∞ (link severed)
```
