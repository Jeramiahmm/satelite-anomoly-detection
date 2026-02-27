# ============================================================================
# Astraea-1 Protocol — Makefile
# ============================================================================
# Common operations for development, testing, and deployment.
#
# Usage:
#   make install      — Install Python dependencies
#   make test         — Run test suite
#   make red-team     — Run all red-team attack simulations
#   make docker-up    — Bootstrap CA + build + launch constellation
#   make docker-down  — Tear down the constellation
#   make dashboard    — Open the live terminal dashboard
#   make clean        — Remove generated artifacts
# ============================================================================

.PHONY: install test red-team lint docker-up docker-down docker-logs \
        dashboard bootstrap-ca clean help

PYTHON ?= python3

# ── Development ──────────────────────────────────────────────────────────────

install:  ## Install all Python dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the full test suite
	$(PYTHON) -m pytest tests/test_integration.py -v --tb=short

lint:  ## Check syntax of all Python files
	$(PYTHON) -m py_compile astraea/node/satellite_node.py
	$(PYTHON) -m py_compile astraea/messaging/isl.py
	$(PYTHON) -m py_compile federated_aggregator.py
	$(PYTHON) -m py_compile scripts/red_team.py
	@echo "All files compile successfully."

# ── Red Team ─────────────────────────────────────────────────────────────────

red-team:  ## Run all three attack simulations and export results
	$(PYTHON) -m scripts.red_team --attack polarized --output results/polarized.json
	$(PYTHON) -m scripts.red_team --attack spacejack --output results/spacejack.json
	$(PYTHON) -m scripts.red_team --attack bitflip --output results/bitflip.json
	@echo ""
	@echo "Results written to results/"
	@ls -la results/*.json

red-team-polarized:  ## Run polarized attack only
	$(PYTHON) -m scripts.red_team --attack polarized

red-team-spacejack:  ## Run spacejack attack only
	$(PYTHON) -m scripts.red_team --attack spacejack

red-team-bitflip:  ## Run bitflip attack only
	$(PYTHON) -m scripts.red_team --attack bitflip

# ── Docker ───────────────────────────────────────────────────────────────────

bootstrap-ca:  ## Generate shared CA and per-node certificates
	$(PYTHON) -m scripts.bootstrap_ca

docker-up: bootstrap-ca  ## Build and launch the 5-node constellation + Grafana
	docker compose up --build -d
	@echo ""
	@echo "Constellation launched!"
	@echo ""
	@echo "  Grafana dashboard:  http://localhost:3000  (admin / astraea)"
	@echo "  Prometheus:         http://localhost:9091"
	@echo "  NATS monitoring:    http://localhost:8222"
	@echo ""
	@echo "  make docker-logs       — Follow all node logs"
	@echo "  make dashboard         — Terminal dashboard"
	@echo "  make docker-red-team   — Run red-team inside container"
	@echo "  make docker-down       — Stop everything"

docker-down:  ## Tear down the constellation
	docker compose down

docker-logs:  ## Follow logs from all nodes
	docker compose logs -f

docker-red-team:  ## Run red-team attack inside sat-03 container
	docker compose exec sat-03 python -m scripts.red_team --attack polarized

docker-status:  ## Check container health status
	docker compose ps

# ── Monitoring ───────────────────────────────────────────────────────────────

dashboard:  ## Open the live terminal dashboard (Docker must be running)
	$(PYTHON) -m scripts.dashboard --docker

dashboard-local:  ## Open dashboard for local (non-Docker) testing
	$(PYTHON) -m scripts.dashboard

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove generated artifacts
	rm -rf certs/ results/ __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	@echo "Cleaned."

# ── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
