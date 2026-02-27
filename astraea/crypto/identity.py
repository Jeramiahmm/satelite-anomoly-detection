"""
SPIFFE/SPIRE-Inspired Identity Manager for Astraea-1.

Generates ephemeral X.509 certificates with SPIFFE-compatible IDs,
manages a local Certificate Revocation List (CRL), and provides
mTLS context factories for inter-satellite link security.

References:
    - SPIFFE: https://spiffe.io/docs/latest/spiffe-about/overview/
    - NIST SP 800-207 §3: Zero Trust Architecture tenets
    - RFC 5280: X.509 PKI Certificate and CRL Profile
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import ssl
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509 import (
    CertificateRevocationListBuilder,
    RevokedCertificateBuilder,
    UniformResourceIdentifier,
)
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants aligned with CCSDS 350.0-G-3 recommendations
# ---------------------------------------------------------------------------
TRUST_DOMAIN = "astraea-1.mesh"
CERT_TTL_SECONDS = 300  # 5-minute ephemeral certs (aggressive rotation)
CA_TTL_DAYS = 30
CURVE = ec.SECP384R1()  # NIST P-384 — Suite B compliant


@dataclass
class NodeIdentity:
    """Represents a satellite node's cryptographic identity."""

    node_id: str
    spiffe_id: str
    private_key: ec.EllipticCurvePrivateKey
    certificate: x509.Certificate
    ca_certificate: x509.Certificate
    cert_path: Optional[Path] = None
    key_path: Optional[Path] = None
    ca_path: Optional[Path] = None


@dataclass
class CertificateAuthority:
    """Constellation-level Certificate Authority (simulated SPIRE server)."""

    private_key: ec.EllipticCurvePrivateKey
    certificate: x509.Certificate
    _crl_serials: set = field(default_factory=set)
    _crl_lock: threading.Lock = field(default_factory=threading.Lock)
    _crl_number: int = field(default=0)

    def is_revoked(self, serial_number: int) -> bool:
        with self._crl_lock:
            return serial_number in self._crl_serials

    def revoke(self, serial_number: int) -> None:
        with self._crl_lock:
            self._crl_serials.add(serial_number)
            self._crl_number += 1
            logger.warning(
                "REVOKED certificate serial=%d  crl_revision=%d",
                serial_number,
                self._crl_number,
            )

    def get_revoked_serials(self) -> frozenset:
        with self._crl_lock:
            return frozenset(self._crl_serials)

    def build_crl_pem(self) -> bytes:
        """Build a DER/PEM-encoded CRL for distribution over ISL."""
        with self._crl_lock:
            builder = CertificateRevocationListBuilder()
            builder = builder.issuer_name(self.certificate.subject)
            builder = builder.last_update(datetime.datetime.now(datetime.timezone.utc))
            builder = builder.next_update(
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=CERT_TTL_SECONDS)
            )
            for serial in self._crl_serials:
                revoked = (
                    RevokedCertificateBuilder()
                    .serial_number(serial)
                    .revocation_date(datetime.datetime.now(datetime.timezone.utc))
                    .build()
                )
                builder = builder.add_revoked_certificate(revoked)
            crl = builder.sign(
                private_key=self.private_key,
                algorithm=hashes.SHA384(),
            )
            return crl.public_bytes(serialization.Encoding.PEM)


def generate_ca() -> CertificateAuthority:
    """Generate a constellation-level CA keypair and self-signed certificate."""
    private_key = ec.generate_private_key(CURVE)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Astraea-1 Constellation"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"CA.{TRUST_DOMAIN}"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=CA_TTL_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(private_key, hashes.SHA384())
    )
    logger.info("Generated constellation CA: %s", cert.subject)
    return CertificateAuthority(private_key=private_key, certificate=cert)


def issue_node_certificate(
    ca: CertificateAuthority,
    node_id: str,
    san_ip: Optional[str] = None,
    ttl_seconds: int = CERT_TTL_SECONDS,
) -> NodeIdentity:
    """
    Issue an ephemeral X.509 certificate for a satellite node.

    The certificate encodes a SPIFFE ID as a URI SAN:
        spiffe://astraea-1.mesh/satellite/{node_id}

    Args:
        ca: The constellation Certificate Authority.
        node_id: Unique satellite identifier (e.g., "sat-01").
        san_ip: Optional IP address to add as a SAN.
        ttl_seconds: Certificate time-to-live in seconds.

    Returns:
        NodeIdentity with private key, certificate, and CA cert.
    """
    spiffe_id = f"spiffe://{TRUST_DOMAIN}/satellite/{node_id}"
    private_key = ec.generate_private_key(CURVE)

    subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Astraea-1 Constellation"),
        x509.NameAttribute(NameOID.COMMON_NAME, node_id),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)

    san_entries: list[x509.GeneralName] = [
        UniformResourceIdentifier(spiffe_id),
        x509.DNSName(node_id),
        x509.DNSName("localhost"),
    ]
    if san_ip:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(san_ip)))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca.certificate.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(seconds=ttl_seconds))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )
        .sign(ca.private_key, hashes.SHA384())
    )

    logger.info(
        "Issued cert for %s  spiffe_id=%s  serial=%d  ttl=%ds",
        node_id,
        spiffe_id,
        cert.serial_number,
        ttl_seconds,
    )

    return NodeIdentity(
        node_id=node_id,
        spiffe_id=spiffe_id,
        private_key=private_key,
        certificate=cert,
        ca_certificate=ca.certificate,
    )


def persist_identity(identity: NodeIdentity, directory: Path) -> NodeIdentity:
    """Write identity materials to disk for TLS context loading."""
    directory.mkdir(parents=True, exist_ok=True)

    cert_path = directory / f"{identity.node_id}.crt"
    key_path = directory / f"{identity.node_id}.key"
    ca_path = directory / "ca.crt"

    cert_path.write_bytes(
        identity.certificate.public_bytes(serialization.Encoding.PEM)
    )
    key_path.write_bytes(
        identity.private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(key_path, 0o600)
    ca_path.write_bytes(
        identity.ca_certificate.public_bytes(serialization.Encoding.PEM)
    )

    identity.cert_path = cert_path
    identity.key_path = key_path
    identity.ca_path = ca_path
    return identity


def create_mtls_server_context(identity: NodeIdentity) -> ssl.SSLContext:
    """
    Build an ssl.SSLContext for the mTLS *server* side.

    Enforces CERT_REQUIRED — every connecting peer must present a valid
    certificate signed by our constellation CA.
    """
    identity = _ensure_persisted(identity)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=str(identity.cert_path), keyfile=str(identity.key_path))
    ctx.load_verify_locations(cafile=str(identity.ca_path))
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False  # We verify SPIFFE ID programmatically
    return ctx


def create_mtls_client_context(identity: NodeIdentity) -> ssl.SSLContext:
    """
    Build an ssl.SSLContext for the mTLS *client* side.

    Presents our certificate to the remote peer and verifies the
    server's certificate against the constellation CA.
    """
    identity = _ensure_persisted(identity)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=str(identity.cert_path), keyfile=str(identity.key_path))
    ctx.load_verify_locations(cafile=str(identity.ca_path))
    ctx.check_hostname = False  # SPIFFE verification instead of hostname
    return ctx


def extract_spiffe_id(cert_pem: bytes) -> Optional[str]:
    """Extract the SPIFFE ID (URI SAN) from a PEM-encoded certificate."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        )
        for uri in san.value.get_values_for_type(UniformResourceIdentifier):
            if uri.startswith("spiffe://"):
                return uri
    except x509.ExtensionNotFound:
        pass
    return None


def _ensure_persisted(identity: NodeIdentity) -> NodeIdentity:
    """Persist identity to a temp directory if not yet on disk."""
    if identity.cert_path is None:
        tmpdir = Path(tempfile.mkdtemp(prefix=f"astraea_{identity.node_id}_"))
        persist_identity(identity, tmpdir)
    return identity
