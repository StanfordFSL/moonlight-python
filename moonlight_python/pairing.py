"""5-stage challenge-response pairing protocol.

Reference: moonlight-qt/app/backend/nvpairingmanager.cpp:190-355
Byte-exact reimplementation of the pairing handshake.
"""

from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives import hashes
from cryptography.x509 import load_pem_x509_certificate

from .identity import Identity
from .http_client import NvHTTP
from .server import ServerInfo
from .exceptions import PairingError, PairingAlreadyInProgress, WrongPinError


def _aes_encrypt_ecb(plaintext: bytes, key: bytes) -> bytes:
    """AES-128-ECB encrypt with no padding (input must be multiple of 16 bytes)."""
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    enc = cipher.encryptor()
    return enc.update(plaintext) + enc.finalize()


def _aes_decrypt_ecb(ciphertext: bytes, key: bytes) -> bytes:
    """AES-128-ECB decrypt with no padding."""
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    dec = cipher.decryptor()
    return dec.update(ciphertext) + dec.finalize()


def _get_hash_algo_and_length(server_major_version: int) -> tuple[str, int]:
    """Gen 7+ uses SHA-256 (32 bytes), older uses SHA-1 (20 bytes)."""
    if server_major_version >= 7:
        return "sha256", 32
    else:
        return "sha1", 20


def _hash_data(algo: str, data: bytes) -> bytes:
    h = hashlib.new(algo)
    h.update(data)
    return h.digest()


def _get_cert_signature(cert_pem: bytes) -> bytes:
    """Extract the raw signature from a PEM certificate."""
    cert = load_pem_x509_certificate(cert_pem)
    return cert.signature


def _verify_signature(data: bytes, signature: bytes, server_cert_pem: bytes) -> bool:
    """Verify RSA-SHA256 signature using the server's certificate public key."""
    cert = load_pem_x509_certificate(server_cert_pem)
    pub_key = cert.public_key()
    try:
        pub_key.verify(  # type: ignore[union-attr]
            signature,
            data,
            asym_padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def _sign_data(identity: Identity, data: bytes) -> bytes:
    """Sign data with RSA-SHA256 using our private key."""
    return identity.private_key.sign(
        data,
        asym_padding.PKCS1v15(),
        hashes.SHA256(),
    )


def pair(http: NvHTTP, identity: Identity, pin: str, server_info: ServerInfo) -> bytes:
    """Execute the 5-stage pairing protocol.

    Args:
        http: NvHTTP client (pre-configured with address)
        identity: Client identity (keypair + cert)
        pin: 4-digit PIN displayed on the server
        server_info: Server info (need app_version for hash algo selection)

    Returns:
        Server certificate PEM bytes (to be pinned for future HTTPS connections)

    Raises:
        PairingError: On protocol failure
        PairingAlreadyInProgress: If server is already in a pairing session
        WrongPinError: If the PIN is incorrect
    """
    # Determine hash algorithm based on server generation
    major_version = 7  # Default to gen 7+ (SHA-256)
    if server_info.app_version:
        parts = server_info.app_version.split(".")
        if parts:
            try:
                major_version = int(parts[0])
            except ValueError:
                pass

    hash_algo, hash_length = _get_hash_algo_and_length(major_version)

    # --- Stage 1: Get server cert ---
    salt = os.urandom(16)
    salted_pin = salt + pin.encode("utf-8")
    aes_key = _hash_data(hash_algo, salted_pin)[:16]

    get_cert_xml = http.open_http(
        "pair",
        f"devicename=roth&updateState=1&phrase=getservercert"
        f"&salt={salt.hex()}&clientcert={identity.cert_pem.hex()}"
    )
    NvHTTP.verify_response_status(get_cert_xml)

    if NvHTTP.get_xml_string(get_cert_xml, "paired") != "1":
        raise PairingError("Failed pairing at stage #1")

    server_cert_hex = NvHTTP.get_xml_string_from_hex(get_cert_xml, "plaincert")
    if server_cert_hex is None:
        # Server likely already pairing
        try:
            http.open_http("unpair")
        except Exception:
            pass
        raise PairingAlreadyInProgress("Server is already in a pairing session")

    server_cert_pem = server_cert_hex  # This is the PEM bytes from hex decoding

    # Pin this cert for HTTPS until pairing completes
    http.server_cert_pem = server_cert_pem

    # --- Stage 2: Client challenge ---
    random_challenge = os.urandom(16)
    encrypted_challenge = _aes_encrypt_ecb(random_challenge, aes_key)

    challenge_xml = http.open_http(
        "pair",
        f"devicename=roth&updateState=1&clientchallenge={encrypted_challenge.hex()}"
    )
    NvHTTP.verify_response_status(challenge_xml)

    if NvHTTP.get_xml_string(challenge_xml, "paired") != "1":
        _unpair_on_failure(http)
        raise PairingError("Failed pairing at stage #2")

    # --- Stage 3: Server challenge response ---
    challenge_response_data_enc = NvHTTP.get_xml_string_from_hex(challenge_xml, "challengeresponse")
    if challenge_response_data_enc is None:
        _unpair_on_failure(http)
        raise PairingError("Missing challengeresponse in stage #2 reply")

    challenge_response_data = _aes_decrypt_ecb(challenge_response_data_enc, aes_key)
    client_secret = os.urandom(16)

    # Server response = first hash_length bytes of decrypted challenge response
    server_response = challenge_response_data[:hash_length]

    # Build our challenge response:
    # [server challenge (16 bytes after hash)] + [client cert signature] + [client secret]
    challenge_response = bytearray()
    challenge_response += challenge_response_data[hash_length:hash_length + 16]
    challenge_response += identity.cert_signature()
    challenge_response += client_secret

    # Hash the challenge response and pad to 32 bytes
    padded_hash = _hash_data(hash_algo, bytes(challenge_response))
    padded_hash = padded_hash.ljust(32, b'\x00')

    encrypted_response_hash = _aes_encrypt_ecb(padded_hash, aes_key)

    resp_xml = http.open_http(
        "pair",
        f"devicename=roth&updateState=1&serverchallengeresp={encrypted_response_hash.hex()}"
    )
    NvHTTP.verify_response_status(resp_xml)

    if NvHTTP.get_xml_string(resp_xml, "paired") != "1":
        _unpair_on_failure(http)
        raise PairingError("Failed pairing at stage #3")

    # Verify server's pairing secret
    pairing_secret = NvHTTP.get_xml_string_from_hex(resp_xml, "pairingsecret")
    if pairing_secret is None:
        _unpair_on_failure(http)
        raise PairingError("Missing pairingsecret in stage #3 reply")

    server_secret = pairing_secret[:16]
    server_signature = pairing_secret[16:]

    # MITM check: verify the server signed its secret with its own cert
    if not _verify_signature(server_secret, server_signature, server_cert_pem):
        _unpair_on_failure(http)
        raise PairingError("MITM detected: server signature verification failed")

    # PIN verification: hash(our_challenge + server_cert_sig + server_secret) must match server_response
    expected_data = random_challenge + _get_cert_signature(server_cert_pem) + server_secret
    expected_response = _hash_data(hash_algo, expected_data)

    if expected_response != server_response:
        _unpair_on_failure(http)
        raise WrongPinError("Incorrect PIN")

    # --- Stage 4: Client pairing secret (sent over HTTPS with pinned cert) ---
    client_pairing_secret = client_secret + _sign_data(identity, client_secret)

    secret_resp_xml = http.open_http(
        "pair",
        f"devicename=roth&updateState=1&clientpairingsecret={client_pairing_secret.hex()}"
    )
    NvHTTP.verify_response_status(secret_resp_xml)

    if NvHTTP.get_xml_string(secret_resp_xml, "paired") != "1":
        _unpair_on_failure(http)
        raise PairingError("Failed pairing at stage #4")

    # --- Stage 5: Pair challenge over HTTPS ---
    pair_challenge_xml = http.open_https(
        "pair",
        "devicename=roth&updateState=1&phrase=pairchallenge"
    )
    NvHTTP.verify_response_status(pair_challenge_xml)

    if NvHTTP.get_xml_string(pair_challenge_xml, "paired") != "1":
        _unpair_on_failure(http)
        raise PairingError("Failed pairing at stage #5")

    return server_cert_pem


def _unpair_on_failure(http: NvHTTP) -> None:
    """Send unpair request to clean up a failed pairing attempt."""
    try:
        http.open_http("unpair")
    except Exception:
        pass
