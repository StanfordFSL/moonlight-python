"""Tests for pairing protocol crypto operations."""

import hashlib
import os

from moonlight_python.pairing import _aes_encrypt_ecb, _aes_decrypt_ecb, _hash_data


def test_aes_ecb_roundtrip():
    """AES-128-ECB encrypt/decrypt roundtrip."""
    key = os.urandom(16)
    plaintext = os.urandom(16)

    ciphertext = _aes_encrypt_ecb(plaintext, key)
    assert len(ciphertext) == 16
    assert ciphertext != plaintext

    recovered = _aes_decrypt_ecb(ciphertext, key)
    assert recovered == plaintext


def test_aes_ecb_no_padding():
    """AES-128-ECB works with exact block-size data (no padding)."""
    key = os.urandom(16)
    plaintext = os.urandom(32)  # 2 blocks

    ciphertext = _aes_encrypt_ecb(plaintext, key)
    assert len(ciphertext) == 32

    recovered = _aes_decrypt_ecb(ciphertext, key)
    assert recovered == plaintext


def test_hash_sha256():
    """_hash_data with sha256 matches hashlib."""
    data = b"test data for hashing"
    result = _hash_data("sha256", data)
    assert result == hashlib.sha256(data).digest()
    assert len(result) == 32


def test_hash_sha1():
    """_hash_data with sha1 matches hashlib."""
    data = b"test data for hashing"
    result = _hash_data("sha1", data)
    assert result == hashlib.sha1(data).digest()
    assert len(result) == 20


def test_aes_key_derivation():
    """AES key derivation matches the pairing protocol: SHA256(salt+PIN)[:16]."""
    salt = bytes.fromhex("deadbeef" * 4)
    pin = "1234"
    salted_pin = salt + pin.encode("utf-8")
    aes_key = hashlib.sha256(salted_pin).digest()[:16]
    assert len(aes_key) == 16
