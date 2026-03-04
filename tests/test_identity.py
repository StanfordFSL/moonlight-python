"""Tests for identity generation and persistence."""

import tempfile
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa

from moonlight_python.identity import Identity


def test_generate_identity():
    """Identity generates RSA 2048 key, X.509 cert, and unique ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ident = Identity(tmpdir)

        # RSA 2048
        assert isinstance(ident.private_key, rsa.RSAPrivateKey)
        assert ident.private_key.key_size == 2048

        # X.509 cert
        cert = ident.certificate
        assert isinstance(cert, x509.Certificate)

        # CN = "NVIDIA GameStream Client"
        cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0]
        assert cn.value == "NVIDIA GameStream Client"

        # Serial number 1 (C++ uses 0 but Python cryptography requires > 0)
        assert cert.serial_number == 1

        # Unique ID is a hex string
        assert len(ident.unique_id) > 0
        int(ident.unique_id, 16)  # Should not raise


def test_identity_persistence():
    """Identity is loaded from disk on second instantiation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ident1 = Identity(tmpdir)
        uid1 = ident1.unique_id
        cert_pem1 = ident1.cert_pem

        ident2 = Identity(tmpdir)
        assert ident2.unique_id == uid1
        assert ident2.cert_pem == cert_pem1


def test_cert_signature_extraction():
    """cert_signature() returns raw bytes from the certificate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ident = Identity(tmpdir)
        sig = ident.cert_signature()
        assert isinstance(sig, bytes)
        assert len(sig) > 0  # RSA 2048 signature is 256 bytes
        assert len(sig) == 256


def test_pem_formats():
    """cert_pem and key_pem produce valid PEM data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ident = Identity(tmpdir)
        assert ident.cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")
        assert ident.key_pem.startswith(b"-----BEGIN RSA PRIVATE KEY-----")
