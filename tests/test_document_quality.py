from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest

from app.core.images.quality import (
    assess_image_quality,
    assess_qr_image_quality,
)


def document(paper: int = 238) -> np.ndarray:
    image = np.full((900, 1400, 3), 45, dtype=np.uint8)
    cv2.rectangle(image, (90, 80), (1310, 820), (paper,) * 3, -1)
    cv2.rectangle(image, (90, 80), (1310, 820), (20,) * 3, 5)
    for row in range(180, 740, 70):
        cv2.putText(
            image,
            f"DOCUMENT TEXT {row}",
            (180, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (25,) * 3,
            3,
            cv2.LINE_AA,
        )
    return image


def encoded(image: np.ndarray) -> BytesIO:
    valid, content = cv2.imencode(".png", image)
    assert valid
    return BytesIO(content.tobytes())


def assess(image: np.ndarray):
    return assess_image_quality(encoded(image))


@pytest.mark.parametrize("noise_sigma", [5, 15, 30])
def test_sensor_noise_does_not_promote_a_blurrier_document(noise_sigma):
    sharp = document(210)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    noise = np.random.default_rng(42).normal(0, noise_sigma, sharp.shape[:2])
    noisy = np.clip(blurred.astype(float) + noise[..., None], 0, 255).astype(
        np.uint8
    )

    better = assess(sharp)
    worse = assess(noisy)

    assert better.document_detected and worse.document_detected
    assert better.score > worse.score
    assert worse.noise_level > better.noise_level


def test_sharp_white_paper_beats_a_dimmed_blurred_copy():
    sharp = document(255)
    blurred = (cv2.GaussianBlur(sharp, (3, 3), 0) * 0.95).astype(np.uint8)

    better = assess(sharp)
    worse = assess(blurred)

    assert better.glare_ratio == 0
    assert better.score > worse.score


def test_sharpness_still_distinguishes_already_sharp_frames():
    sharp = document()
    mildly_blurred = cv2.GaussianBlur(sharp, (3, 3), 0.5)

    better = assess(sharp)
    worse = assess(mildly_blurred)

    assert (
        better.component_scores["sharpness"]
        > worse.component_scores["sharpness"]
    )
    assert better.score > worse.score


def test_rotated_document_focus_is_not_improved_by_textured_background():
    image = np.full((1000, 1200, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (200, 220), (1000, 780), (220,) * 3, -1)
    cv2.rectangle(image, (200, 220), (1000, 780), (20,) * 3, 5)
    for row in range(300, 760, 60):
        cv2.putText(
            image,
            f"DOCUMENT TEXT {row}",
            (250, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (25,) * 3,
            3,
            cv2.LINE_AA,
        )
    rotation = cv2.getRotationMatrix2D((600, 500), 20, 1)
    sharp = cv2.warpAffine(
        image, rotation, (1200, 1000), borderValue=(80,) * 3
    )
    protected = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(protected, (190, 210), (1010, 790), 255, -1)
    protected = cv2.warpAffine(protected, rotation, (1200, 1000)) > 0
    blurred = cv2.GaussianBlur(sharp, (5, 5), 0)
    background = np.clip(
        80 + np.random.default_rng(42).normal(0, 8, image.shape[:2]), 0, 255
    ).astype(np.uint8)
    blurred[~protected] = background[~protected, None]

    better = assess(sharp)
    worse = assess(blurred)

    assert better.document_detected and worse.document_detected
    assert better.sharpness_variance > worse.sharpness_variance
    assert better.score > worse.score


def test_localized_glare_includes_the_blank_center_of_the_reflection():
    sharp = document()
    reflection = sharp.copy()
    cv2.circle(reflection, (480, 450), 155, (255,) * 3, -1)

    better = assess(sharp)
    worse = assess(reflection)

    assert worse.glare_ratio > 0.07
    assert better.score > worse.score


def test_document_detail_is_preferred_to_a_sharp_thumbnail():
    sharp = document()
    thumbnail = cv2.resize(sharp, (700, 450), interpolation=cv2.INTER_AREA)

    better = assess(sharp)
    worse = assess(thumbnail)

    assert (
        better.component_scores["resolution"]
        > worse.component_scores["resolution"]
    )
    assert better.score > worse.score


def test_underexposure_reduces_document_quality():
    sharp = document()
    dark = (sharp * 0.4).astype(np.uint8)

    better = assess(sharp)
    worse = assess(dark)

    assert (
        better.component_scores["exposure"]
        > worse.component_scores["exposure"]
    )
    assert better.score > worse.score


def test_qr_selection_retains_its_existing_measurements():
    from app.modules.qr.selection import assess_image_quality as qr_assessor

    assert qr_assessor is assess_qr_image_quality
    source = encoded(document(255))
    qr = qr_assessor(source)
    assert source.tell() == 0
    document_quality = assess_image_quality(source)
    assert source.tell() == 0
    assert qr.glare_ratio > 0.04
    assert document_quality.glare_ratio == 0


@pytest.mark.parametrize("size", [(1, 1), (8, 3), (30, 30)])
def test_tiny_or_featureless_images_have_finite_bounded_scores(size):
    quality = assess(np.full((*size, 3), 255, dtype=np.uint8))

    assert np.isfinite(quality.score)
    assert 0 <= quality.score <= 1
    assert all(0 <= score <= 1 for score in quality.component_scores.values())
