#!/bin/bash
# Astraea-1 Satellite Node Entrypoint
# =====================================
# Applies traffic shaping (tc) to simulate orbital dynamics before
# launching the satellite node process.
#
# Environment variables:
#   ASTRAEA_LATENCY_MS    — Base link latency in ms (default: 200)
#   ASTRAEA_JITTER_MS     — Latency jitter in ms (default: 50)
#   ASTRAEA_LOSS_PERCENT  — Packet loss % for grey-hole sim (default: 0)
#   ASTRAEA_BANDWIDTH     — Link bandwidth limit (default: 1mbit)

set -e

LATENCY="${ASTRAEA_LATENCY_MS:-200}"
JITTER="${ASTRAEA_JITTER_MS:-50}"
LOSS="${ASTRAEA_LOSS_PERCENT:-0}"
BANDWIDTH="${ASTRAEA_BANDWIDTH:-1mbit}"
IFACE="eth0"

echo "=============================================="
echo "  ASTRAEA-1 SATELLITE NODE: ${ASTRAEA_NODE_ID}"
echo "=============================================="
echo "  Latency:    ${LATENCY}ms ± ${JITTER}ms"
echo "  Loss:       ${LOSS}%"
echo "  Bandwidth:  ${BANDWIDTH}"
echo "=============================================="

# Apply traffic shaping to simulate orbital dynamics
# Uses HTB (Hierarchical Token Bucket) + netem for realistic modeling
if [ "${LATENCY}" -gt 0 ] 2>/dev/null; then
    echo "[tc] Configuring traffic shaping on ${IFACE}..."

    # Root qdisc: HTB for bandwidth control
    tc qdisc add dev "${IFACE}" root handle 1: htb default 10 2>/dev/null || \
        tc qdisc replace dev "${IFACE}" root handle 1: htb default 10

    # Class: bandwidth limit simulating ISL capacity
    tc class add dev "${IFACE}" parent 1: classid 1:10 htb \
        rate "${BANDWIDTH}" burst 15k 2>/dev/null || \
        tc class replace dev "${IFACE}" parent 1: classid 1:10 htb \
        rate "${BANDWIDTH}" burst 15k

    # Leaf qdisc: netem for latency, jitter, and loss
    NETEM_OPTS="delay ${LATENCY}ms ${JITTER}ms distribution normal"
    if [ "${LOSS}" -gt 0 ] 2>/dev/null; then
        NETEM_OPTS="${NETEM_OPTS} loss ${LOSS}%"
        echo "[tc] Grey-hole simulation: ${LOSS}% packet loss"
    fi

    tc qdisc add dev "${IFACE}" parent 1:10 handle 10: netem ${NETEM_OPTS} 2>/dev/null || \
        tc qdisc replace dev "${IFACE}" parent 1:10 handle 10: netem ${NETEM_OPTS}

    echo "[tc] Traffic shaping applied successfully"
else
    echo "[tc] Skipping traffic shaping (LATENCY=0)"
fi

echo "[boot] Starting satellite node process..."
exec "$@"
