# Astraea-1 Protocol — Satellite Node Container
# ================================================
# Multi-stage build for minimal attack surface.
# Uses iproute2/tc for orbital dynamics traffic shaping.

FROM python:3.11-slim AS base

# Install traffic shaping tools (tc) for orbital simulation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        iptables \
        iputils-ping \
        net-tools \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY astraea/ ./astraea/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Make entrypoint executable
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ASTRAEA_NODE_ID=sat-01
ENV ASTRAEA_NATS_URL=nats://nats-server:4222
ENV ASTRAEA_CERT_DIR=/tmp/astraea/certs

EXPOSE 8443
EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "astraea.node.satellite_node"]
