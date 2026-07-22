from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.ocr_engine.mrz.factory import create_mrz_scanner
from app.ocr_engine.mrz.recognizer import MrzScanResult, recognize_mrz

POLYGON = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], dtype=np.float32)


class MrzScannerEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((40, 120, 3), dtype=np.uint8)

    def test_factory_selects_cpu_two_stage_with_lazy_imports(self) -> None:
        backend = object()
        model_type = object()
        constructor = Mock()
        with patch.dict(
            sys.modules,
            {
                "capybara": SimpleNamespace(
                    Backend=SimpleNamespace(cpu=backend)
                ),
                "mrzscanner": SimpleNamespace(
                    MRZScanner=constructor,
                    ModelType=SimpleNamespace(two_stage=model_type),
                ),
            },
        ):
            scanner = create_mrz_scanner()

        constructor.assert_called_once_with(
            backend=backend, model_type=model_type
        )
        self.assertIs(scanner, constructor.return_value)

    def test_preserves_all_raw_rows_and_disables_scanner_normalization(
        self,
    ) -> None:
        texts = ["  p<4RMTEST<<N4ME< ", "AA1234<  ", ""]
        scanner = Mock(
            return_value={"mrz_polygon": POLYGON, "mrz_texts": texts}
        )

        result = recognize_mrz(self.image, scanner)

        self.assertEqual(result, MrzScanResult(tuple(texts), 1))
        scanner.assert_called_once_with(
            self.image, do_center_crop=False, do_postprocess=False
        )

    def test_handles_absent_detection_and_empty_recognition(self) -> None:
        for output, expected in (
            (None, MrzScanResult((), 0)),
            ({"mrz_polygon": None, "mrz_texts": None}, MrzScanResult((), 0)),
            ({"mrz_polygon": [], "mrz_texts": []}, MrzScanResult((), 0)),
            (
                {"mrz_polygon": [], "mrz_texts": ["STALE"]},
                MrzScanResult((), 0),
            ),
            ({"mrz_polygon": POLYGON, "mrz_texts": []}, MrzScanResult((), 1)),
        ):
            with self.subTest(output=output):
                self.assertEqual(
                    recognize_mrz(self.image, Mock(return_value=output)),
                    expected,
                )

    def test_rejects_malformed_model_output(self) -> None:
        for output in (
            [],
            {},
            {"mrz_polygon": POLYGON, "mrz_texts": "P<ARM"},
            {"mrz_polygon": POLYGON, "mrz_texts": ["P<ARM", 123]},
            {"mrz_polygon": "invalid", "mrz_texts": []},
            {"mrz_polygon": [[0, 0], [1, 1]], "mrz_texts": []},
            {"mrz_polygon": np.full((4, 2), np.nan), "mrz_texts": []},
        ):
            with self.subTest(output=output):
                with self.assertRaises(RuntimeError):
                    recognize_mrz(self.image, Mock(return_value=output))

    def test_native_value_error_is_a_recognition_failure(self) -> None:
        error = ValueError("Input polygon does not contain four points")

        with self.assertRaisesRegex(
            RuntimeError, "MRZScanner recognition failed"
        ) as raised:
            recognize_mrz(self.image, Mock(side_effect=error))

        self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
