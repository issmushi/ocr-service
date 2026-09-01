from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.quality import ImageQualityAssessment
from app.modules.qr.selection import (
    QRImageCandidate,
    prepare_qr_candidates,
)
from app.modules.qr.service import decode_qr


def assessment(
    *,
    sharpness: float,
    contrast: float,
) -> ImageQualityAssessment:
    return ImageQualityAssessment(
        score=0.5,
        grade="medium",
        width=1_200,
        height=800,
        document_detected=False,
        touches_frame=False,
        document_area_ratio=0.0,
        border_margin_ratio=0.0,
        sharpness_variance=200.0,
        glare_ratio=0.0,
        brightness=150.0,
        underexposure_ratio=0.0,
        overexposure_ratio=0.0,
        illumination_spread=20.0,
        local_contrast=20.0,
        noise_level=2.0,
        perspective_distortion=0.0,
        component_scores={
            "document_completeness": 0.0,
            "sharpness": sharpness,
            "glare": 1.0,
            "exposure": 1.0,
            "perspective": 0.0,
            "local_contrast": contrast,
            "resolution": 1.0,
            "noise_and_blur": sharpness,
        },
    )


def candidate(index: int, score: float) -> QRImageCandidate:
    return QRImageCandidate(
        image=UploadFile(
            filename=f"qr-{index + 1}.png",
            file=BytesIO(b"image"),
        ),
        index=index,
        size=5,
        quality_score=score,
        assessment=assessment(sharpness=score, contrast=score),
    )


class QRServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ranks_qr_frames_by_qr_specific_quality(self) -> None:
        uploads = [
            UploadFile(filename="qr-1.png", file=BytesIO(b"first")),
            UploadFile(filename="qr-2.png", file=BytesIO(b"second")),
            UploadFile(filename="qr-3.png", file=BytesIO(b"third")),
        ]
        with patch(
            "app.modules.qr.selection.assess_image_quality",
            side_effect=[
                assessment(sharpness=0.3, contrast=0.4),
                assessment(sharpness=0.9, contrast=0.8),
                assessment(sharpness=0.6, contrast=0.7),
            ],
        ):
            result = await prepare_qr_candidates(uploads)

        self.assertEqual([item.index for item in result], [1, 2, 0])

    async def test_rejects_more_than_configured_number_of_frames(self) -> None:
        uploads = [
            UploadFile(filename=f"qr-{index}.png", file=BytesIO(b"image"))
            for index in range(settings.qr_max_images_per_request + 1)
        ]

        with self.assertRaises(AppError) as error_context:
            await prepare_qr_candidates(uploads)

        self.assertEqual(
            error_context.exception.code,
            ErrorCode.VALIDATION_ERROR,
        )

    async def test_tries_next_ranked_frame_when_qr_is_not_found(self) -> None:
        candidates = [candidate(1, 0.9), candidate(0, 0.6)]
        expected = {
            "success": True,
            "data": {"passport": {"surname": "ТЕСТОВ"}},
            "error": None,
        }
        decode_candidate = AsyncMock(
            side_effect=[
                AppError(ErrorCode.QR_NOT_FOUND, "QR-код не найден"),
                expected,
            ]
        )
        with (
            patch(
                "app.modules.qr.service.prepare_qr_candidates",
                new=AsyncMock(return_value=candidates),
            ),
            patch(
                "app.modules.qr.service.decode_candidate",
                new=decode_candidate,
            ),
        ):
            result = await decode_qr([item.image for item in candidates])

        self.assertEqual(result, expected)
        self.assertEqual(decode_candidate.await_count, 2)
        self.assertIs(
            decode_candidate.await_args_list[0].args[0], candidates[0]
        )
        self.assertIs(
            decode_candidate.await_args_list[1].args[0], candidates[1]
        )

    async def test_does_not_retry_when_qr_service_is_unavailable(self) -> None:
        candidates = [candidate(0, 0.9), candidate(1, 0.8)]
        decode_candidate = AsyncMock(
            side_effect=AppError(
                ErrorCode.QR_SERVICE_UNAVAILABLE,
                "QR-модуль недоступен",
            )
        )
        with (
            patch(
                "app.modules.qr.service.prepare_qr_candidates",
                new=AsyncMock(return_value=candidates),
            ),
            patch(
                "app.modules.qr.service.decode_candidate",
                new=decode_candidate,
            ),
        ):
            with self.assertRaises(AppError) as error_context:
                await decode_qr([item.image for item in candidates])

        self.assertEqual(
            error_context.exception.code,
            ErrorCode.QR_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(decode_candidate.await_count, 1)

    async def test_returns_invalid_payload_if_no_frame_is_valid(self) -> None:
        candidates = [candidate(0, 0.9), candidate(1, 0.8)]
        decode_candidate = AsyncMock(
            side_effect=[
                AppError(ErrorCode.QR_NOT_FOUND, "QR-код не найден"),
                AppError(
                    ErrorCode.QR_INVALID_PAYLOAD,
                    "QR-код содержит некорректные данные",
                ),
            ]
        )
        with (
            patch(
                "app.modules.qr.service.prepare_qr_candidates",
                new=AsyncMock(return_value=candidates),
            ),
            patch(
                "app.modules.qr.service.decode_candidate",
                new=decode_candidate,
            ),
        ):
            with self.assertRaises(AppError) as error_context:
                await decode_qr([item.image for item in candidates])

        self.assertEqual(
            error_context.exception.code,
            ErrorCode.QR_INVALID_PAYLOAD,
        )


if __name__ == "__main__":
    unittest.main()
