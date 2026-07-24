from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.modules.paddle.inference_worker import paddle_inference_worker
from app.ocr_engine.paddle.factory import create_mrz_ocr


class DocumentOrientationDispatchTests(unittest.TestCase):
    @patch("app.ocr_engine.paddle.factory.create_paddle_ocr_quietly")
    def test_mrz_engine_uses_mobile_english_models(self, create_mock) -> None:
        create_mrz_ocr("cpu")

        kwargs = create_mock.call_args.args[0]
        self.assertEqual(
            kwargs["text_detection_model_name"], "PP-OCRv5_mobile_det"
        )
        self.assertEqual(
            kwargs["text_recognition_model_name"],
            "latin_PP-OCRv5_mobile_rec",
        )
        self.assertFalse(kwargs["use_doc_unwarping"])
        self.assertFalse(kwargs["use_textline_orientation"])
        self.assertEqual(kwargs["device"], "cpu")

    def test_worker_decodes_png_and_reuses_mrz_engine_for_orientation(
        self,
    ) -> None:
        image = np.full((13, 19, 3), (11, 77, 201), dtype=np.uint8)
        image[2:6, 4:12] = (211, 33, 9)
        images = (image, np.flip(image, axis=0).copy())
        contents = []
        for original in images:
            encoded, content = cv2.imencode(".png", original)
            self.assertTrue(encoded)
            contents.append(content.tobytes())

        general, mrz = object(), object()
        connection = object()
        results = []

        def serve(connection_arg, initialize, dispatch):
            self.assertIs(connection_arg, connection)
            engine = initialize()
            self.assertIs(engine["general"], general)
            self.assertIs(engine["mrz"], mrz)
            for content in contents:
                results.append(
                    dispatch(engine, "document_orientation", (content,), {})
                )

        with (
            patch(
                "app.modules.paddle.inference_worker.serve_inference_worker",
                side_effect=serve,
            ) as serve_worker,
            patch(
                "app.ocr_engine.paddle.factory.create_ocr",
                return_value=general,
            ) as create_general,
            patch(
                "app.ocr_engine.paddle.factory.create_mrz_ocr",
                return_value=mrz,
            ) as create_mrz,
            patch(
                "app.ocr_engine.paddle.orientation.predict_document_orientation",
                side_effect=[
                    SimpleNamespace(rotation_degrees=90, confidence=0.97),
                    SimpleNamespace(rotation_degrees=270, confidence=0.81),
                ],
            ) as predict,
        ):
            paddle_inference_worker(connection, "cpu")

        serve_worker.assert_called_once()
        create_general.assert_called_once_with(device="cpu")
        create_mrz.assert_called_once_with(device="cpu")
        self.assertEqual(predict.call_count, 2)
        for request, original in zip(predict.call_args_list, images):
            self.assertEqual(len(request.args), 1)
            self.assertIsInstance(request.args[0], np.ndarray)
            self.assertEqual(request.args[0].dtype, np.dtype(np.uint8))
            np.testing.assert_array_equal(request.args[0], original)
            self.assertEqual(set(request.kwargs), {"ocr"})
            self.assertIs(request.kwargs["ocr"], mrz)
        self.assertEqual(
            results,
            [
                {"rotation_degrees": 90, "confidence": 0.97},
                {"rotation_degrees": 270, "confidence": 0.81},
            ],
        )


if __name__ == "__main__":
    unittest.main()
