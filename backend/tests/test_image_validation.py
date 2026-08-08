"""Upload validation tests (ROUTES.md flaw #2).

The truncation case is the important one: project_context.md §2.9 bug #5
records that PIL's Image.verify() accepts truncated JPEGs, which is why the
validator does a full decode instead. This test is what keeps someone from
"optimizing" it back to verify().
"""

from __future__ import annotations

import pytest

from app.services.image_validation import ImageValidationError, validate_image


def test_accepts_valid_jpeg(jpeg_bytes):
    result = validate_image(jpeg_bytes, "image/jpeg")
    assert result.width == 256
    assert result.height == 256
    assert result.size_bytes == len(jpeg_bytes)


def test_rejects_empty_upload():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(b"", "image/jpeg")


def test_rejects_non_image():
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        validate_image(b"%PDF-1.4 this is a pdf", "image/jpeg")


def test_rejects_disallowed_content_type(jpeg_bytes):
    with pytest.raises(ImageValidationError, match="unsupported content type"):
        validate_image(jpeg_bytes, "application/pdf")


def test_rejects_too_small(tiny_jpeg_bytes):
    with pytest.raises(ImageValidationError, match="smallest side"):
        validate_image(tiny_jpeg_bytes, "image/jpeg")


def test_rejects_oversized(jpeg_bytes, monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(
        "app.services.image_validation.get_settings",
        lambda: Settings(max_upload_bytes=10),
    )
    with pytest.raises(ImageValidationError, match="limit is"):
        validate_image(jpeg_bytes, "image/jpeg")


def test_rejects_truncated_jpeg(jpeg_bytes):
    """A truncated JPEG must be rejected.

    Image.verify() passes these; only a full decode catches them.
    """
    truncated = jpeg_bytes[: len(jpeg_bytes) // 2]
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        validate_image(truncated, "image/jpeg")
