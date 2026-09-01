from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.qr.crypto import decrypt_json_payload, encrypt_json_payload
from app.modules.qr.schemas import QRGenerateRequest
from app.modules.qr.service import decode_qr_content, generate_qr_png


def crypto_settings() -> SimpleNamespace:
    encoded_key = (
        base64.urlsafe_b64encode(b"k" * 32).decode("ascii").rstrip("=")
    )
    return SimpleNamespace(
        qr_active_key_id="test",
        qr_encryption_keys=f"test:{encoded_key}",
    )


class QRCryptoRoundTripTests(unittest.TestCase):
    def test_compact_token_round_trip(self) -> None:
        payload = {"passport": {"surname": "ТЕСТОВ"}}

        with patch("app.modules.qr.crypto.settings", new=crypto_settings()):
            token = encrypt_json_payload(payload)
            decoded, key_id = decrypt_json_payload(token)

        self.assertTrue(token.startswith("OQ2.test."))
        self.assertEqual(decoded, payload)
        self.assertEqual(key_id, "test")

    def test_generated_qr_can_be_decoded_locally(self) -> None:
        payload = QRGenerateRequest(
            surname="ТЕСТОВ",
            name="ТЕСТ",
            patronymic="ТЕСТОВИЧ",
            sex="МУЖ.",
            date_of_birth="15.01.1990",
            passport_series="45 15",
            passport_number="123456",
        )

        with patch("app.modules.qr.crypto.settings", new=crypto_settings()):
            image = generate_qr_png(payload)
            response = decode_qr_content(
                image,
                filename="qr.png",
                content_type="image/png",
            )

        self.assertTrue(response["success"])
        self.assertEqual(response["data"]["passport"]["surname"], "ТЕСТОВ")
        self.assertEqual(response["data"]["meta"]["key_id"], "test")


if __name__ == "__main__":
    unittest.main()
