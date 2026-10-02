from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.modules.russian.inference_worker import russian_inference_worker
from app.modules.russian.passport import recognize_passport_text_bytes
from tests.russian.test_engine_adapter import image_bytes


class InferenceWorkerTests(unittest.TestCase):
    def test_full_text_reuses_pipeline_without_passport_processing(
        self,
    ) -> None:
        pipeline = Mock()
        pipeline.words_detector.predict.return_value = {
            "WordsDetector": {"bbox": [[2, 2, 15, 10, 0.9, 0, "Word"]]}
        }
        pipeline.ocr_cyr.predict.return_value = {
            "OCRCyrillic": {"ocr_output": "ПРИМЕР"}
        }
        with (
            patch(
                "app.ocr_engine.russian.factory.create_pipeline",
                return_value=pipeline,
            ) as factory,
            patch(
                "app.modules.russian.inference_worker.serve_inference_worker"
            ) as serve,
        ):
            russian_inference_worker(Mock(), "cpu")
            _, initialize, dispatch = serve.call_args.args
            reader = initialize()
            for _ in range(2):
                result = dispatch(reader, "full_text", (image_bytes(),), {})
                self.assertEqual(result, {"text": "ПРИМЕР"})

        factory.assert_called_once_with(device="cpu")
        self.assertEqual(pipeline.words_detector.predict.call_count, 2)
        self.assertEqual(pipeline.ocr_cyr.predict.call_count, 2)
        pipeline.process_img.assert_not_called()
        pipeline.ocr_cyr.fix_errors.assert_not_called()

    def test_dispatch_reuses_one_pipeline_for_passport_page(self) -> None:
        result = SimpleNamespace(
            ocr={
                "Last_name_en": "EXAMPLE",
                "First_name_en": "IVAN IVANOVICH",
                "Birth_date": "02.03.1990",
                "Licence_number": "12 34 567890",
            },
            doctype="DrivingLicense",
            timings={"total": 0.5},
        )
        pipeline = Mock()
        pipeline.process_img.return_value = result
        connection = Mock()
        with (
            patch(
                "app.ocr_engine.russian.factory.create_pipeline",
                return_value=pipeline,
            ) as factory,
            patch(
                "app.modules.russian.inference_worker.serve_inference_worker",
            ) as serve,
        ):
            russian_inference_worker(connection, "cpu")
            actual_connection, initialize, dispatch = serve.call_args.args
            self.assertIs(actual_connection, connection)
            self.assertIs(initialize(), pipeline)
            factory.assert_called_once_with(device="cpu")
            content = image_bytes()
            self.assertEqual(
                dispatch(
                    pipeline,
                    "passport_text",
                    (content,),
                    {"suffix": ".png"},
                ),
                recognize_passport_text_bytes(content, pipeline=pipeline),
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported RussianDocsOCR operation"
            ):
                dispatch(pipeline, "unknown", (content,), {})
            with self.assertRaises(ValueError):
                dispatch(pipeline, "passport_text", (b"invalid",), {})
