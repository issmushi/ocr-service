from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.modules.russian.passport import (
    normalize_date,
    normalize_passport_text_result,
    parse_internal_passport_number,
    recognize_passport_text_bytes,
)
from tests.russian.test_engine_adapter import image_bytes


class PassportTests(unittest.TestCase):
    def test_internal_passport_number_requires_exact_four_plus_six_digits(
        self,
    ) -> None:
        self.assertEqual(
            parse_internal_passport_number("39 14 745905"),
            ("3914", "745905"),
        )
        self.assertEqual(
            parse_internal_passport_number("14 745905"), (None, None)
        )

    def test_date_validation_preserves_previous_normalization(self) -> None:
        for value in ("02.03.1990", "1990-03-02", "02031990"):
            self.assertEqual(normalize_date(value), "1990-03-02")
        self.assertIsNone(normalize_date("31.02.1990"))
        self.assertEqual(normalize_date("unparsed"), "unparsed")

    def test_text_adapter_returns_nonempty_raw_fields_and_image_dimensions(
        self,
    ) -> None:
        raw = {
            "Last_name_ru": "  ПРИМЕРОВ  ",
            "First_name_ru": "ИВАН",
            "Middle_name_ru": None,
            "Licence_number": "12 34 567890",
            "Issue_organization_ru": "  ОТДЕЛОМ МВД РОССИИ  ",
        }
        result = recognize_passport_text_bytes(
            image_bytes(),
            Mock(
                process_img=Mock(
                    return_value=SimpleNamespace(
                        ocr=raw, doctype="INTPassport_2011"
                    )
                )
            ),
        )

        self.assertEqual(
            result,
            {
                "items": [
                    {
                        "text": "ПРИМЕРОВ",
                        "confidence": None,
                        "box": None,
                        "detection_confidence": None,
                    },
                    {
                        "text": "ИВАН",
                        "confidence": None,
                        "box": None,
                        "detection_confidence": None,
                    },
                    {
                        "text": "12 34 567890",
                        "confidence": None,
                        "box": None,
                        "detection_confidence": None,
                    },
                    {
                        "text": "ОТДЕЛОМ МВД РОССИИ",
                        "confidence": None,
                        "box": None,
                        "detection_confidence": None,
                    },
                ],
                "image_width": 48,
                "image_height": 32,
                "mrz": None,
                "passport": {
                    "passport_type": "internal",
                    "document_code": "PN",
                    "issuing_country": "RUS",
                    "surname": "ПРИМЕРОВ",
                    "name": "ИВАН",
                    "patronymic": None,
                    "sex": None,
                    "date_of_birth": None,
                    "passport_series": "1234",
                    "passport_number": "567890",
                    "nationality": "RUS",
                    "issue_date": None,
                    "expiry_date": None,
                    "department_code": None,
                    "Issue_organization_ru": "ОТДЕЛОМ МВД РОССИИ",
                    "personal_number": None,
                    "mrz_valid": None,
                },
            },
        )

    def test_text_adapter_preserves_russian_ocr_mrz_rows(self) -> None:
        lines = [
            "PNRUSTESTOV<<TEST".ljust(44, "<"),
            "4511234569RUS9001158M<<<<<<<5150620770001<88",
        ]
        result = recognize_passport_text_bytes(
            image_bytes(),
            Mock(
                process_img=Mock(
                    return_value=SimpleNamespace(
                        ocr={"MRZ": "\n".join(lines)},
                        doctype="INTPassport_2011",
                    )
                )
            ),
        )

        self.assertEqual(result["mrz"], lines)

    def test_text_adapter_maps_external_passport_fields_without_inventing_mrz(
        self,
    ) -> None:
        result = normalize_passport_text_result(
            {
                "Last_name_en": "IVANOV",
                "First_name_en": "IVAN",
                "Middle_name_en": "IVANOVICH",
                "Sex_en": "M",
                "Birth_date": "02.03.1990",
                "Licence_number": "12 3456789",
                "Issue_date": "04.05.2020",
                "Expiration_date": "04.05.2030",
                "Issue_organisation_code": "  FMS  ",
            },
            doctype="EXTPassport_2013",
        )

        self.assertEqual(
            result,
            {
                "passport_type": "international",
                "document_code": "P",
                "issuing_country": "RUS",
                "surname": "IVANOV",
                "name": "IVAN",
                "patronymic": "IVANOVICH",
                "sex": "M",
                "date_of_birth": "1990-03-02",
                "passport_series": None,
                "passport_number": "123456789",
                "nationality": "RUS",
                "issue_date": "2020-05-04",
                "expiry_date": "2030-05-04",
                "department_code": "FMS",
                "Issue_organization_ru": None,
                "personal_number": None,
                "mrz_valid": None,
            },
        )

    def test_text_adapter_does_not_infer_passport_kind_from_ocr_fields(
        self,
    ) -> None:
        result = normalize_passport_text_result(
            {
                "Last_name_ru": "ИВАНОВ",
                "Licence_number": "12 34 567890",
            },
            doctype="Passport",
        )

        self.assertIsNone(result["passport_type"])
        self.assertIsNone(result["document_code"])
        self.assertIsNone(result["issuing_country"])
        self.assertIsNone(result["nationality"])
