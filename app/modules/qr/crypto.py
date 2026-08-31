from __future__ import annotations

import base64
import binascii
import json
import os
import re
import zlib
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings
from app.core.errors import AppError, ErrorCode

TOKEN_TYPE = "ocr.qr.encrypted"
LEGACY_TOKEN_VERSION = 1
TOKEN_VERSION = 2
TOKEN_ALGORITHM = "A256GCM"
TOKEN_PREFIX = "OQ2"
NONCE_SIZE_BYTES = 12
AUTH_TAG_SIZE_BYTES = 16
MAX_TOKEN_LENGTH = 8_192
MAX_PLAINTEXT_BYTES = 32_768
KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _configuration_error() -> AppError:
    return AppError(
        ErrorCode.QR_SERVICE_UNAVAILABLE,
        "Ключ шифрования QR не настроен",
    )


def _invalid_payload_error() -> AppError:
    return AppError(
        ErrorCode.QR_INVALID_PAYLOAD,
        "QR-код содержит повреждённые или неподдерживаемые данные",
    )


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_base64url(
    value: object,
    *,
    expected_length: int | None = None,
) -> bytes:
    if not isinstance(value, str) or not value:
        raise _invalid_payload_error()

    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError, TypeError) as error:
        raise _invalid_payload_error() from error

    if expected_length is not None and len(decoded) != expected_length:
        raise _invalid_payload_error()
    return decoded


def _decode_configured_key(value: str) -> bytes:
    try:
        decoded = _decode_base64url(value, expected_length=32)
    except AppError as error:
        raise _configuration_error() from error
    return decoded


def load_keyring() -> tuple[str, dict[str, bytes]]:
    active_key_id = settings.qr_active_key_id.strip()
    raw_keyring = settings.qr_encryption_keys.strip()
    if not KEY_ID_PATTERN.fullmatch(active_key_id) or not raw_keyring:
        raise _configuration_error()

    keyring: dict[str, bytes] = {}
    for raw_entry in raw_keyring.split(","):
        key_id, separator, encoded_key = raw_entry.strip().partition(":")
        if (
            separator != ":"
            or not KEY_ID_PATTERN.fullmatch(key_id)
            or not encoded_key
            or key_id in keyring
        ):
            raise _configuration_error()
        keyring[key_id] = _decode_configured_key(encoded_key)

    if active_key_id not in keyring:
        raise _configuration_error()
    return active_key_id, keyring


def active_encryption_key_id() -> str:
    active_key_id, _keyring = load_keyring()
    return active_key_id


def _authenticated_data(key_id: str, version: int) -> bytes:
    return (f"{TOKEN_TYPE}|{version}|{TOKEN_ALGORITHM}|{key_id}").encode(
        "ascii"
    )


def encrypt_json_payload(payload: dict[str, Any]) -> str:
    active_key_id, keyring = load_keyring()
    plaintext = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    compressed = zlib.compress(plaintext, level=9)
    nonce = os.urandom(NONCE_SIZE_BYTES)
    encrypted = AESGCM(keyring[active_key_id]).encrypt(
        nonce,
        compressed,
        _authenticated_data(active_key_id, TOKEN_VERSION),
    )
    return ".".join(
        (
            TOKEN_PREFIX,
            active_key_id,
            _encode_base64url(nonce),
            _encode_base64url(encrypted),
        )
    )


def _decode_json_payload(plaintext: bytes) -> dict[str, Any]:
    if len(plaintext) > MAX_PLAINTEXT_BYTES:
        raise _invalid_payload_error()

    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _invalid_payload_error() from error
    if not isinstance(payload, dict):
        raise _invalid_payload_error()
    return payload


def _decompress_payload(compressed: bytes) -> bytes:
    try:
        decompressor = zlib.decompressobj()
        plaintext = decompressor.decompress(
            compressed,
            MAX_PLAINTEXT_BYTES + 1,
        )
        if (
            len(plaintext) > MAX_PLAINTEXT_BYTES
            or not decompressor.eof
            or decompressor.unconsumed_tail
        ):
            raise _invalid_payload_error()
        return plaintext
    except zlib.error as error:
        raise _invalid_payload_error() from error


def _decrypt_compact_token(token: str) -> tuple[dict[str, Any], str]:
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != TOKEN_PREFIX:
        raise _invalid_payload_error()

    _prefix, key_id, encoded_nonce, encoded_encrypted = parts
    if not KEY_ID_PATTERN.fullmatch(key_id):
        raise _invalid_payload_error()

    _active_key_id, keyring = load_keyring()
    key = keyring.get(key_id)
    if key is None:
        raise _invalid_payload_error()

    nonce = _decode_base64url(
        encoded_nonce,
        expected_length=NONCE_SIZE_BYTES,
    )
    encrypted = _decode_base64url(encoded_encrypted)
    if len(encrypted) <= AUTH_TAG_SIZE_BYTES:
        raise _invalid_payload_error()

    try:
        compressed = AESGCM(key).decrypt(
            nonce,
            encrypted,
            _authenticated_data(key_id, TOKEN_VERSION),
        )
    except (InvalidTag, ValueError) as error:
        raise _invalid_payload_error() from error

    return _decode_json_payload(_decompress_payload(compressed)), key_id


def _decrypt_legacy_token(token: str) -> tuple[dict[str, Any], str]:
    try:
        envelope = json.loads(token)
    except (json.JSONDecodeError, TypeError) as error:
        raise _invalid_payload_error() from error

    if not isinstance(envelope, dict):
        raise _invalid_payload_error()

    key_id = envelope.get("key_id")
    if (
        envelope.get("type") != TOKEN_TYPE
        or envelope.get("version") != LEGACY_TOKEN_VERSION
        or envelope.get("algorithm") != TOKEN_ALGORITHM
        or not isinstance(key_id, str)
        or not KEY_ID_PATTERN.fullmatch(key_id)
    ):
        raise _invalid_payload_error()

    _active_key_id, keyring = load_keyring()
    key = keyring.get(key_id)
    if key is None:
        raise _invalid_payload_error()

    nonce = _decode_base64url(
        envelope.get("nonce"),
        expected_length=NONCE_SIZE_BYTES,
    )
    ciphertext = _decode_base64url(envelope.get("ciphertext"))
    auth_tag = _decode_base64url(
        envelope.get("auth_tag"),
        expected_length=AUTH_TAG_SIZE_BYTES,
    )
    if not ciphertext:
        raise _invalid_payload_error()

    try:
        plaintext = AESGCM(key).decrypt(
            nonce,
            ciphertext + auth_tag,
            _authenticated_data(key_id, LEGACY_TOKEN_VERSION),
        )
    except (InvalidTag, ValueError) as error:
        raise _invalid_payload_error() from error

    return _decode_json_payload(plaintext), key_id


def decrypt_json_payload(token: str) -> tuple[dict[str, Any], str]:
    if not token or len(token) > MAX_TOKEN_LENGTH:
        raise _invalid_payload_error()
    if token.startswith(f"{TOKEN_PREFIX}."):
        return _decrypt_compact_token(token)
    return _decrypt_legacy_token(token)
