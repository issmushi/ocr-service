from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.images.quality import (
    ImageQualityAssessment,
    assess_image_quality,
)
from app.core.images.selection import (
    SelectedImage,
    select_best_stored_image,
    store_and_select_image_uploads,
)
from app.core.images.uploads import StoredImage


def assessment(score: float, sharpness: float) -> ImageQualityAssessment:
    return ImageQualityAssessment(
        score=score,
        grade="good" if score >= 0.75 else "medium",
        width=1600,
        height=1000,
        document_detected=True,
        touches_frame=False,
        document_area_ratio=0.7,
        border_margin_ratio=0.03,
        sharpness_variance=300.0,
        glare_ratio=0.0,
        brightness=150.0,
        underexposure_ratio=0.0,
        overexposure_ratio=0.0,
        illumination_spread=20.0,
        local_contrast=20.0,
        noise_level=2.0,
        perspective_distortion=0.05,
        component_scores={
            "document_completeness": 1.0,
            "sharpness": sharpness,
            "glare": 1.0,
            "exposure": 1.0,
            "perspective": 1.0,
            "local_contrast": 1.0,
            "resolution": 1.0,
            "noise_and_blur": 1.0,
        },
    )


def stored_image(index: int) -> StoredImage:
    return StoredImage(
        path=Path(f"/tmp/frame-{index}.jpg"),
        filename=f"frame-{index}.jpg",
        content_type="image/jpeg",
        suffix=".jpg",
        size=100,
    )


class ImageSelectionTests(unittest.IsolatedAsyncioTestCase):
    def test_quality_prefers_sharp_document_over_blurred_copy(self) -> None:
        image = np.full((900, 1_400, 3), 45, dtype=np.uint8)
        cv2.rectangle(image, (90, 80), (1_310, 820), (238, 238, 238), -1)
        cv2.rectangle(image, (90, 80), (1_310, 820), (20, 20, 20), 5)
        for row in range(180, 740, 70):
            cv2.putText(
                image,
                f"DOCUMENT TEXT {row}",
                (180, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (25, 25, 25),
                3,
                cv2.LINE_AA,
            )
        blurred = cv2.GaussianBlur(image, (31, 31), 0)

        with TemporaryDirectory() as directory:
            sharp_path = Path(directory) / "sharp.png"
            blurred_path = Path(directory) / "blurred.png"
            cv2.imwrite(str(sharp_path), image)
            cv2.imwrite(str(blurred_path), blurred)

            sharp_quality = assess_image_quality(sharp_path)
            blurred_quality = assess_image_quality(blurred_path)

        self.assertTrue(sharp_quality.document_detected)
        self.assertGreater(
            sharp_quality.sharpness_variance,
            blurred_quality.sharpness_variance,
        )
        self.assertGreater(sharp_quality.score, blurred_quality.score)

    def test_selects_candidate_with_highest_quality_score(self) -> None:
        images = [stored_image(1), stored_image(2), stored_image(3)]
        with patch(
            "app.core.images.selection.assess_image_quality",
            side_effect=[
                assessment(0.60, 0.70),
                assessment(0.90, 0.80),
                assessment(0.75, 0.95),
            ],
        ):
            selected = select_best_stored_image(images)

        self.assertEqual(selected.index, 1)
        self.assertEqual(selected.image, images[1])
        self.assertEqual(selected.candidate_count, 3)

    async def test_removes_all_unselected_images(self) -> None:
        uploads = [
            UploadFile(filename=f"frame-{index}.jpg", file=BytesIO(b"image"))
            for index in range(1, 4)
        ]
        stored = [stored_image(1), stored_image(2), stored_image(3)]
        selected = SelectedImage(
            image=stored[1],
            index=1,
            assessment=assessment(0.9, 0.9),
            candidate_count=3,
        )

        with (
            patch(
                "app.core.images.selection.store_image_upload",
                new=AsyncMock(side_effect=stored),
            ),
            patch(
                "app.core.images.selection.select_best_stored_image",
                return_value=selected,
            ),
            patch("app.core.images.selection.delete_upload") as delete_upload,
        ):
            result = await store_and_select_image_uploads(
                uploads,
                "job-id",
                "document.jpg",
            )

        self.assertEqual(result, selected)
        self.assertEqual(
            [call.args[0] for call in delete_upload.call_args_list],
            [str(stored[0].path), str(stored[2].path)],
        )

    async def test_removes_saved_candidates_when_later_upload_fails(
        self,
    ) -> None:
        uploads = [
            UploadFile(filename="frame-1.jpg", file=BytesIO(b"image")),
            UploadFile(filename="frame-2.jpg", file=BytesIO(b"image")),
        ]
        first_stored_image = stored_image(1)

        with (
            patch(
                "app.core.images.selection.store_image_upload",
                new=AsyncMock(
                    side_effect=[
                        first_stored_image,
                        AppError(
                            ErrorCode.IMAGE_READ_ERROR,
                            "Изображение невозможно прочитать",
                        ),
                    ]
                ),
            ),
            patch("app.core.images.selection.delete_upload") as delete_upload,
        ):
            with self.assertRaises(AppError):
                await store_and_select_image_uploads(
                    uploads,
                    "job-id",
                    "document.jpg",
                )

        delete_upload.assert_called_once_with(str(first_stored_image.path))

    async def test_rejects_more_than_configured_number_of_images(self) -> None:
        uploads = [
            UploadFile(filename=f"frame-{index}.jpg", file=BytesIO(b"image"))
            for index in range(settings.ocr_max_images_per_request + 1)
        ]

        with self.assertRaises(AppError) as error_context:
            await store_and_select_image_uploads(
                uploads,
                "job-id",
                "document.jpg",
            )

        self.assertEqual(
            error_context.exception.code,
            ErrorCode.VALIDATION_ERROR,
        )


if __name__ == "__main__":
    unittest.main()
