"""Firebase Storage access for diagnosis images.

Images are stored under `<prefix>/<uid>/<diagnosis_id>` so an object's path
itself carries ownership — a worker or admin tool can't accidentally serve
one farmer's photo under another's record.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.firebase import get_bucket

log = logging.getLogger(__name__)


def object_name(uid: str, diagnosis_id: str) -> str:
    return f"{get_settings().storage_image_prefix}/{uid}/{diagnosis_id}"


def upload_image(uid: str, diagnosis_id: str, data: bytes, content_type: str) -> str:
    """Store the original upload. Returns the object name."""
    name = object_name(uid, diagnosis_id)
    blob = get_bucket().blob(name)
    blob.upload_from_string(data, content_type=content_type)
    log.info("stored image %s (%d bytes)", name, len(data))
    return name


def download_image(name: str) -> bytes:
    blob = get_bucket().blob(name)
    if not blob.exists():
        raise FileNotFoundError(f"image object not found: {name}")
    return blob.download_as_bytes()


def delete_image(name: str) -> bool:
    """Delete an image; returns False if it was already gone."""
    blob = get_bucket().blob(name)
    try:
        blob.delete()
        return True
    except Exception as exc:
        log.warning("could not delete image %s: %s", name, exc)
        return False
