"""
Integration tests for the Astraea-1 Protocol.

Tests the full detection-to-isolation pipeline without Docker or NATS,
verifying that:
    1. VAE can detect anomalous telemetry
    2. PDP correctly revokes compromised nodes
    3. CRL propagates and blocks revoked certificates
    4. Routing mesh isolates untrusted nodes
    5. Federation excludes revoked nodes from aggregation
"""

import time

import torch
import pytest

from astraea.crypto.identity import (
    generate_ca,
    issue_node_certificate,
    extract_spiffe_id,
    TRUST_DOMAIN,
)
from astraea.crypto.crl_manager import CRLCache
from astraea.ml.vae import TelemetryVAE, reconstruction_error, vae_loss
from astraea.ml.anomaly_detector import AnomalyDetector
from astraea.ml.telemetry import (
    AttackMode,
    generate_telemetry_batch,
    normalize_telemetry,
)
from astraea.ml.trainer import LocalTrainer
from astraea.federation.aggregator import FederatedAggregator
from astraea.routing.link_state import LinkStateRouter
from astraea.policy.pdp import PolicyDecisionPoint, PolicyConfig, TrustLevel


class TestCryptoIdentity:
    """Tests for the SPIFFE-inspired identity layer."""

    def test_ca_generation(self):
        ca = generate_ca()
        assert ca.private_key is not None
        assert ca.certificate is not None
        assert "Astraea-1" in ca.certificate.subject.rfc4514_string()

    def test_node_certificate_issuance(self):
        ca = generate_ca()
        identity = issue_node_certificate(ca, "sat-01")
        assert identity.node_id == "sat-01"
        assert identity.spiffe_id == f"spiffe://{TRUST_DOMAIN}/satellite/sat-01"
        assert identity.certificate.serial_number > 0

    def test_spiffe_id_extraction(self):
        from cryptography.hazmat.primitives import serialization

        ca = generate_ca()
        identity = issue_node_certificate(ca, "sat-test")
        pem = identity.certificate.public_bytes(serialization.Encoding.PEM)
        spiffe_id = extract_spiffe_id(pem)
        assert spiffe_id == f"spiffe://{TRUST_DOMAIN}/satellite/sat-test"

    def test_crl_revocation(self):
        ca = generate_ca()
        identity = issue_node_certificate(ca, "sat-compromised")
        serial = identity.certificate.serial_number

        assert not ca.is_revoked(serial)
        ca.revoke(serial)
        assert ca.is_revoked(serial)

    def test_crl_pem_generation(self):
        ca = generate_ca()
        identity = issue_node_certificate(ca, "sat-01")
        ca.revoke(identity.certificate.serial_number)
        crl_pem = ca.build_crl_pem()
        assert b"-----BEGIN X509 CRL-----" in crl_pem


class TestCRLCache:
    """Tests for the distributed CRL cache."""

    def test_add_and_check(self):
        crl = CRLCache()
        assert crl.add_revocation(12345)
        assert crl.is_revoked(12345)
        assert not crl.is_revoked(99999)

    def test_duplicate_add(self):
        crl = CRLCache()
        assert crl.add_revocation(111)
        assert not crl.add_revocation(111)  # Already cached

    def test_serialize_deserialize(self):
        crl = CRLCache()
        crl.add_revocation(100)
        crl.add_revocation(200)
        data = crl.serialize()
        deserialized = CRLCache.deserialize_broadcast(data)
        assert 100 in deserialized
        assert 200 in deserialized

    def test_merge_from_broadcast(self):
        crl1 = CRLCache()
        crl1.add_revocation(10)
        crl1.add_revocation(20)

        crl2 = CRLCache()
        added = crl2.merge_from_broadcast(crl1.get_all_revoked())
        assert added == 2
        assert crl2.is_revoked(10)
        assert crl2.is_revoked(20)

    def test_revocation_listener(self):
        crl = CRLCache()
        events = []
        crl.on_revocation(lambda serial: events.append(serial))
        crl.add_revocation(555)
        assert 555 in events


class TestVAE:
    """Tests for the Variational Autoencoder."""

    def test_forward_shape(self):
        model = TelemetryVAE(input_dim=3, latent_dim=8)
        x = torch.randn(16, 3)
        x_hat, mu, log_var = model(x)
        assert x_hat.shape == (16, 3)
        assert mu.shape == (16, 8)
        assert log_var.shape == (16, 8)

    def test_reconstruct_shape(self):
        model = TelemetryVAE()
        x = torch.randn(4, 3)
        x_hat = model.reconstruct(x)
        assert x_hat.shape == (4, 3)

    def test_reconstruction_error(self):
        model = TelemetryVAE()
        x = torch.randn(8, 3)
        errors = reconstruction_error(model, x)
        assert errors.shape == (8,)
        assert (errors >= 0).all()

    def test_vae_loss(self):
        model = TelemetryVAE()
        x = torch.randn(8, 3)
        x_hat, mu, log_var = model(x)
        loss = vae_loss(x, x_hat, mu, log_var)
        assert loss.item() > 0
        assert loss.requires_grad


class TestAnomalyDetector:
    """Tests for the dynamic anomaly detection engine."""

    def _make_trained_detector(self):
        model = TelemetryVAE()
        trainer = LocalTrainer(model)
        trainer.snapshot_global_weights()
        for _ in range(5):
            trainer.train_epoch(batch_size=64, num_batches=20)

        detector = AnomalyDetector(model, warmup_samples=50)
        # Warmup
        for t in range(60):
            raw = generate_telemetry_batch(1, AttackMode.NOMINAL, time_step=t)
            x = normalize_telemetry(raw).squeeze(0)
            detector.ingest(x)
        return detector

    def test_nominal_not_anomalous(self):
        detector = self._make_trained_detector()
        anomaly_count = 0
        for t in range(20):
            raw = generate_telemetry_batch(1, AttackMode.NOMINAL, time_step=100 + t)
            x = normalize_telemetry(raw).squeeze(0)
            state = detector.ingest(x)
            if state.is_anomalous:
                anomaly_count += 1
        # Allow at most 2 spurious detections out of 20 nominal samples
        # (VAE reconstruction error is stochastic)
        assert anomaly_count <= 2, f"Too many false positives: {anomaly_count}/20"

    def test_spacejack_detected(self):
        detector = self._make_trained_detector()
        detected = False
        for t in range(100):
            raw = generate_telemetry_batch(
                1, AttackMode.SPACEJACK, time_step=100 + t, attack_intensity=1.0
            )
            x = normalize_telemetry(raw).squeeze(0)
            state = detector.ingest(x)
            if state.is_anomalous:
                detected = True
                break
        assert detected, "Spacejack attack should be detected"


class TestFederation:
    """Tests for the federated aggregation engine."""

    def test_submit_and_aggregate(self):
        model = TelemetryVAE()
        agg = FederatedAggregator(model, min_participants=2)

        # Two nodes submit deltas
        delta1 = {k: torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}
        delta2 = {k: torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}

        agg.submit_delta("sat-01", delta1)
        agg.submit_delta("sat-02", delta2)

        result = agg.aggregate()
        assert result is not None
        assert agg.round_id == 1

    def test_revoked_node_rejected(self):
        model = TelemetryVAE()
        agg = FederatedAggregator(model, min_participants=1)

        delta = {k: torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}
        accepted = agg.submit_delta(
            "sat-bad",
            delta,
            revoked_serials=frozenset({12345}),
            node_serial=12345,
        )
        assert not accepted

    def test_insufficient_participants(self):
        model = TelemetryVAE()
        agg = FederatedAggregator(model, min_participants=3)

        delta = {k: torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}
        agg.submit_delta("sat-01", delta)

        result = agg.aggregate()
        assert result is None  # Not enough participants

    def test_gradient_clipping(self):
        model = TelemetryVAE()
        agg = FederatedAggregator(model, min_participants=1, clip_norm=1.0)

        # Very large delta should be clipped
        large_delta = {k: torch.ones_like(v) * 100.0 for k, v in model.state_dict().items()}
        agg.submit_delta("sat-01", large_delta)

        result = agg.aggregate()
        assert result is not None


class TestRouting:
    """Tests for the trust-aware link-state routing."""

    def test_basic_routing(self):
        router = LinkStateRouter("sat-01")
        router.add_link("sat-01", "sat-02", 200.0)
        router.add_link("sat-02", "sat-03", 300.0)
        router.add_link("sat-01", "sat-03", 800.0)
        router.recompute_routes()

        # Direct path to sat-02
        assert router.get_next_hop("sat-02") == "sat-02"
        # Via sat-02 to sat-03 (200+300=500 < 800)
        assert router.get_next_hop("sat-03") == "sat-02"

    def test_node_isolation(self):
        router = LinkStateRouter("sat-01")
        router.add_link("sat-01", "sat-02", 200.0)
        router.add_link("sat-02", "sat-03", 200.0)
        router.add_link("sat-01", "sat-03", 500.0)
        router.recompute_routes()

        # Before isolation: route to sat-03 goes via sat-02
        assert router.get_next_hop("sat-03") == "sat-02"

        # Isolate sat-02
        router.isolate_node("sat-02")

        # After isolation: direct path to sat-03
        assert router.get_next_hop("sat-03") == "sat-03"
        # sat-02 is unreachable
        assert router.get_next_hop("sat-02") is None

    def test_trust_score_affects_routing(self):
        router = LinkStateRouter("sat-01")
        router.add_link("sat-01", "sat-02", 200.0, trust_score=1.0)
        router.add_link("sat-01", "sat-03", 300.0, trust_score=1.0)
        router.add_link("sat-02", "sat-04", 200.0, trust_score=1.0)
        router.add_link("sat-03", "sat-04", 200.0, trust_score=1.0)
        router.recompute_routes()

        # Default: route via sat-02 (200+200=400 < 300+200=500)
        assert router.get_next_hop("sat-04") == "sat-02"

        # Degrade sat-02 trust → cost becomes 200/0.2 + 200 = 1200
        router.update_trust_score("sat-02", 0.2)

        # Now route via sat-03 (300+200=500 < 1200)
        assert router.get_next_hop("sat-04") == "sat-03"

    def test_lsa_generation_and_application(self):
        router1 = LinkStateRouter("sat-01")
        router1.add_link("sat-01", "sat-02", 200.0)
        router1.add_link("sat-01", "sat-03", 300.0)

        lsa = router1.generate_lsa()
        assert lsa["origin"] == "sat-01"
        assert "sat-02" in lsa["neighbors"]

        # Another router applies the LSA
        router2 = LinkStateRouter("sat-02")
        router2.add_link("sat-02", "sat-01", 200.0)
        router2.apply_lsa(lsa)

        # Now router2 knows about sat-01 → sat-03
        reachable = router2.get_reachable_nodes()
        assert "sat-01" in reachable


class TestPolicyDecisionPoint:
    """Tests for the PDP policy engine."""

    def test_normal_score_stays_trusted(self):
        pdp = PolicyDecisionPoint(config=PolicyConfig())
        pdp.register_node("sat-01", cert_serial=1000)
        state = pdp.evaluate("sat-01", anomaly_score=0.1)
        assert state.trust_level == TrustLevel.TRUSTED
        assert not state.revoked

    def test_high_score_triggers_revocation(self):
        revoked_nodes = []
        pdp = PolicyDecisionPoint(
            config=PolicyConfig(consecutive_required=2),
            on_revoke=lambda nid, serial: revoked_nodes.append((nid, serial)),
        )
        pdp.register_node("sat-bad", cert_serial=2000)

        # First high score: not revoked yet (need consecutive)
        pdp.evaluate("sat-bad", anomaly_score=0.9)
        assert len(revoked_nodes) == 0

        # Second consecutive high score: REVOKED
        state = pdp.evaluate("sat-bad", anomaly_score=0.95)
        assert state.trust_level == TrustLevel.REVOKED
        assert state.revoked
        assert (("sat-bad", 2000)) in revoked_nodes

    def test_degraded_trust_level(self):
        pdp = PolicyDecisionPoint(config=PolicyConfig())
        pdp.register_node("sat-01", cert_serial=1000)
        state = pdp.evaluate("sat-01", anomaly_score=0.5)
        assert state.trust_level == TrustLevel.DEGRADED

    def test_revoked_nodes_list(self):
        pdp = PolicyDecisionPoint(
            config=PolicyConfig(consecutive_required=1),
        )
        pdp.register_node("sat-01", cert_serial=1)
        pdp.register_node("sat-02", cert_serial=2)

        pdp.evaluate("sat-01", 0.95)  # Revoked
        pdp.evaluate("sat-02", 0.1)   # Normal

        assert "sat-01" in pdp.get_revoked_nodes()
        assert "sat-02" in pdp.get_trusted_nodes()


class TestEndToEndPipeline:
    """
    Full pipeline test: telemetry → detection → revocation → isolation.

    This verifies the < 2 second SLA for the detection-to-isolation path.
    """

    def test_full_detection_to_isolation(self):
        # Setup
        ca = generate_ca()
        target_id = "sat-target"
        identity = issue_node_certificate(ca, target_id)
        serial = identity.certificate.serial_number

        # ML
        model = TelemetryVAE()
        trainer = LocalTrainer(model)
        trainer.snapshot_global_weights()
        for _ in range(5):
            trainer.train_epoch(batch_size=64, num_batches=20)

        detector = AnomalyDetector(model, warmup_samples=50)
        for t in range(60):
            raw = generate_telemetry_batch(1, AttackMode.NOMINAL, time_step=t)
            x = normalize_telemetry(raw).squeeze(0)
            detector.ingest(x)

        # Routing
        router = LinkStateRouter("sat-01")
        router.add_link("sat-01", target_id, 300.0)
        router.add_link(target_id, "sat-01", 300.0)
        router.recompute_routes()
        assert router.get_next_hop(target_id) is not None

        # CRL
        crl = CRLCache()

        # PDP
        isolated = []
        revoked = []

        def on_revoke(nid, cert_serial):
            revoked.append(cert_serial)
            crl.add_revocation(cert_serial)

        def on_isolate(nid):
            isolated.append(nid)
            router.isolate_node(nid)

        pdp = PolicyDecisionPoint(
            config=PolicyConfig(consecutive_required=2),
            on_revoke=on_revoke,
            on_isolate=on_isolate,
        )
        pdp.register_node(target_id, serial)

        # Attack
        start = time.time()
        for t in range(500):
            raw = generate_telemetry_batch(
                1, AttackMode.SPACEJACK, time_step=100 + t, attack_intensity=1.0
            )
            x = normalize_telemetry(raw).squeeze(0)
            state = detector.ingest(x)
            pdp.evaluate(target_id, state.last_normalized_score)

            if crl.is_revoked(serial):
                break

        elapsed = time.time() - start

        # Assertions
        assert crl.is_revoked(serial), "Certificate should be revoked"
        assert target_id in isolated, "Node should be isolated"
        assert router.get_next_hop(target_id) is None, "Node should be unreachable"
        assert elapsed < 2.0, f"Detection-to-isolation took {elapsed:.3f}s (SLA: < 2.0s)"
