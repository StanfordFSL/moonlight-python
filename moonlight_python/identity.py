"""RSA 2048 keypair + X.509 cert generation and persistence.

Reference: moonlight-qt/app/backend/identitymanager.cpp
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


class Identity:
    """Manages the client identity (RSA keypair, X.509 cert, unique ID)."""

    def __init__(self, config_dir: str | Path = "~/.moonlight-python"):
        self.config_dir = Path(config_dir).expanduser()
        self._key_path = self.config_dir / "key.pem"
        self._cert_path = self.config_dir / "cert.pem"
        self._uid_path = self.config_dir / "unique_id"

        self._private_key: rsa.RSAPrivateKey | None = None
        self._cert: x509.Certificate | None = None
        self._unique_id: str | None = None

        self._load_or_generate()

    def _load_or_generate(self) -> None:
        if self._key_path.exists() and self._cert_path.exists() and self._uid_path.exists():
            self._private_key = serialization.load_pem_private_key(
                self._key_path.read_bytes(), password=None
            )
            self._cert = x509.load_pem_x509_certificate(self._cert_path.read_bytes())
            self._unique_id = self._uid_path.read_text().strip()
        else:
            self._generate()

    def _generate(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # RSA 2048 keypair
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Self-signed X.509v3 cert matching identitymanager.cpp
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "NVIDIA GameStream Client"),
        ])
        now = datetime.now(timezone.utc)
        self._cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self._private_key.public_key())
            .serial_number(0)
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=365 * 20))
            .sign(self._private_key, hashes.SHA256())
        )

        # Unique ID: random 64-bit hex string
        uid_bytes = secrets.token_bytes(8)
        self._unique_id = int.from_bytes(uid_bytes, "big").__format__("x")

        # Persist
        self._key_path.write_bytes(
            self._private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        self._cert_path.write_bytes(self._cert.public_bytes(serialization.Encoding.PEM))
        self._uid_path.write_text(self._unique_id)

    @property
    def private_key(self) -> rsa.RSAPrivateKey:
        assert self._private_key is not None
        return self._private_key

    @property
    def certificate(self) -> x509.Certificate:
        assert self._cert is not None
        return self._cert

    @property
    def unique_id(self) -> str:
        assert self._unique_id is not None
        return self._unique_id

    @property
    def cert_pem(self) -> bytes:
        """Certificate as PEM bytes."""
        return self.certificate.public_bytes(serialization.Encoding.PEM)

    @property
    def cert_der(self) -> bytes:
        """Certificate as DER bytes."""
        return self.certificate.public_bytes(serialization.Encoding.DER)

    @property
    def key_pem(self) -> bytes:
        """Private key as PEM bytes."""
        return self.private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )

    def cert_signature(self) -> bytes:
        """Extract the raw signature bytes from the certificate.

        Reference: nvpairingmanager.cpp getSignatureFromPemCert()
        """
        return self.certificate.signature
