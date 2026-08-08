"""Firestore access for the model-version registry.

Firestore holds *which* version is active and when each was registered; the
weights themselves live on disk under MODEL_REGISTRY_DIR. Keeping the pointer
in the database is what lets POST /admin/models/{version}/activate work
without a redeploy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud import firestore as gcf

from app.firebase import COLLECTION_MODEL_VERSIONS, get_db

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_active_version_name() -> str | None:
    q = (
        get_db()
        .collection(COLLECTION_MODEL_VERSIONS)
        .where(filter=gcf.FieldFilter("active", "==", True))
        .limit(1)
    )
    for snap in q.stream():
        return snap.to_dict().get("version")
    return None


def get_record(version: str) -> dict | None:
    snap = get_db().collection(COLLECTION_MODEL_VERSIONS).document(version).get()
    return snap.to_dict() if snap.exists else None


def list_records() -> list[dict]:
    return [
        snap.to_dict()
        for snap in get_db().collection(COLLECTION_MODEL_VERSIONS).stream()
    ]


def register(version: str, metrics: dict, notes: str | None = None) -> dict:
    """Record a version as available. Does not activate it."""
    existing = get_record(version) or {}
    doc = {
        "version": version,
        "metrics": metrics,
        "notes": notes,
        "active": existing.get("active", False),
        "registered_at": existing.get("registered_at") or _now(),
        "updated_at": _now(),
    }
    get_db().collection(COLLECTION_MODEL_VERSIONS).document(version).set(doc)
    return doc


def activate(version: str) -> None:
    """Make `version` the only active one, atomically.

    Batched so there is never a moment with two active versions (or zero) —
    a reader resolving the active version mid-switch would otherwise get an
    arbitrary answer.
    """
    db = get_db()
    batch = db.batch()
    for snap in (
        db.collection(COLLECTION_MODEL_VERSIONS)
        .where(filter=gcf.FieldFilter("active", "==", True))
        .stream()
    ):
        if snap.id != version:
            batch.update(snap.reference, {"active": False, "updated_at": _now()})
    batch.set(
        db.collection(COLLECTION_MODEL_VERSIONS).document(version),
        {"version": version, "active": True, "updated_at": _now()},
        merge=True,
    )
    batch.commit()
    log.info("activated model version %s", version)
