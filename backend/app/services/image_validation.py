"""Upload validation — ROUTES.md flaw #2.

Rejecting a bad upload synchronously (status `rejected`) is much better than
letting it fail deep inside a Celery task, where the farmer waits for a poll
cycle to learn their photo was a PDF.

The decode check here is deliberately a FULL decode, not `Image.verify()`.
project_context.md §2.9 bug #5 records that `verify()` passes truncated
JPEGs — the training pipeline hit exactly this and had to switch to a real
`.convert("RGB").load()`. The same trap applies to uploads, so the same fix
applies here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.config import get_settings


class ImageValidationError(ValueError):
    """Upload is not a usable photo. Maps to status `rejected`."""


@dataclass(frozen=True)
class ValidatedImage:
    content_type: str
    width: int
    height: int
    size_bytes: int


def validate_image(data: bytes, declared_content_type: str | None) -> ValidatedImage:
    """Validate an uploaded photo, or raise ImageValidationError.

    Checks, in cheapest-first order: size, declared type, real decodability,
    dimensions.
    """
    settings = get_settings()

    if not data:
        raise ImageValidationError("uploaded file is empty")

    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"image is {len(data) / (1024 * 1024):.1f} MB; limit is {limit_mb:.0f} MB"
        )

    if declared_content_type and declared_content_type.lower() not in settings.allowed_image_types:
        raise ImageValidationError(
            f"unsupported content type {declared_content_type!r}; "
            f"allowed: {', '.join(settings.allowed_image_types)}"
        )

    # Full decode. verify() would accept truncated JPEGs (see module docstring).
    try:
        with Image.open(io.BytesIO(data)) as im:
            detected_format = (im.format or "").upper()
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
            rgb.load()
            width, height = rgb.size
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(
            "file could not be decoded as an image (it may be corrupt or truncated)"
        ) from exc

    if detected_format not in {"JPEG", "PNG", "WEBP", "MPO"}:
        raise ImageValidationError(
            f"unsupported image format {detected_format or 'unknown'!r}"
        )

    smallest = min(width, height)
    if smallest < settings.min_image_dimension:
        raise ImageValidationError(
            f"image is {width}x{height}; smallest side must be at least "
            f"{settings.min_image_dimension}px for a usable diagnosis"
        )

    return ValidatedImage(
        content_type=declared_content_type or f"image/{detected_format.lower()}",
        width=width,
        height=height,
        size_bytes=len(data),
    )
