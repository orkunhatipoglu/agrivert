"""Firestore access for diagnosis records.

One document per diagnosis in `diagnoses`, keyed by the diagnosis id. The
record carries `owner_uid` so every read can be ownership-checked without a
second lookup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore as gcf

from app.firebase import COLLECTION_DIAGNOSES, COLLECTION_FEEDBACK, get_db
from app.schemas.diagnoses import DiagnosisStatus

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create(
    diagnosis_id: str,
    owner_uid: str,
    image_object: str,
    plot_id: str | None,
    farm_id: str | None,
    image_meta: dict[str, Any],
) -> dict:
    doc = {
        "diagnosis_id": diagnosis_id,
        "owner_uid": owner_uid,
        "status": DiagnosisStatus.QUEUED.value,
        "image_object": image_object,
        "plot_id": plot_id,
        "farm_id": farm_id,
        "image": image_meta,
        "created_at": _now(),
        "updated_at": _now(),
    }
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).set(doc)
    return doc


def create_rejected(
    diagnosis_id: str,
    owner_uid: str,
    reason: str,
    plot_id: str | None,
    farm_id: str | None,
) -> dict:
    """Record an upload that failed validation (flaw #2).

    Persisted rather than discarded so a farmer whose photos keep bouncing
    leaves a trail worth looking at.
    """
    doc = {
        "diagnosis_id": diagnosis_id,
        "owner_uid": owner_uid,
        "status": DiagnosisStatus.REJECTED.value,
        "error": reason,
        "plot_id": plot_id,
        "farm_id": farm_id,
        "created_at": _now(),
        "updated_at": _now(),
    }
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).set(doc)
    return doc


def get(diagnosis_id: str) -> dict | None:
    snap = get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).get()
    return snap.to_dict() if snap.exists else None


def mark_processing(diagnosis_id: str) -> None:
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).update(
        {"status": DiagnosisStatus.PROCESSING.value, "updated_at": _now()}
    )


def save_result(diagnosis_id: str, result: dict) -> None:
    """Write a finished verdict.

    `result` is predict.py's dict; its `status` is already "completed" or
    "uncertain" and is trusted as-is — the threshold decision belongs to the
    model layer, not here.
    """
    payload = {
        "status": result["status"],
        "crop": result.get("crop"),
        "condition": result.get("condition"),
        "healthy": result.get("healthy"),
        "raw_label": result.get("raw_label"),
        "confidence": result.get("confidence"),
        "threshold": result.get("threshold"),
        "field_validated": result.get("field_validated"),
        "alternatives": result.get("alternatives", []),
        "model_version": result.get("model_version"),
        "model_name": result.get("model_name"),
        "updated_at": _now(),
        "completed_at": _now(),
    }
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).update(payload)


def mark_failed(diagnosis_id: str, error: str) -> None:
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).update(
        {
            "status": DiagnosisStatus.FAILED.value,
            "error": error,
            "updated_at": _now(),
        }
    )


def list_for_user(
    owner_uid: str,
    farm_id: str | None = None,
    plot_id: str | None = None,
    status: str | None = None,
    disease_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 50,
) -> list[dict]:
    """History query for GET /diagnoses.

    Composite indexes are required in Firestore for these filter
    combinations; see backend/README.md.
    """
    q = get_db().collection(COLLECTION_DIAGNOSES).where(
        filter=gcf.FieldFilter("owner_uid", "==", owner_uid)
    )
    if farm_id:
        q = q.where(filter=gcf.FieldFilter("farm_id", "==", farm_id))
    if plot_id:
        q = q.where(filter=gcf.FieldFilter("plot_id", "==", plot_id))
    if status:
        q = q.where(filter=gcf.FieldFilter("status", "==", status))
    if disease_id:
        q = q.where(filter=gcf.FieldFilter("raw_label", "==", disease_id))
    if date_from:
        q = q.where(filter=gcf.FieldFilter("created_at", ">=", date_from))
    if date_to:
        q = q.where(filter=gcf.FieldFilter("created_at", "<=", date_to))

    q = q.order_by("created_at", direction=gcf.Query.DESCENDING).limit(limit)
    return [snap.to_dict() for snap in q.stream()]


def delete(diagnosis_id: str) -> None:
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).delete()


def save_feedback(
    diagnosis_id: str,
    owner_uid: str,
    agrees: bool,
    corrected_raw_label: str | None,
    note: str | None,
    predicted_raw_label: str | None,
    model_version: str | None,
    image_object: str | None,
) -> dict:
    """Record farmer feedback (flaw #7).

    Denormalizes the prediction and model version onto the feedback doc: this
    collection is the retraining corpus, and it has to stay interpretable
    after the diagnosis is deleted or the active model moves on.
    """
    doc = {
        "diagnosis_id": diagnosis_id,
        "owner_uid": owner_uid,
        "agrees": agrees,
        "corrected_raw_label": corrected_raw_label,
        "note": note,
        "predicted_raw_label": predicted_raw_label,
        "model_version": model_version,
        "image_object": image_object,
        "created_at": _now(),
    }
    get_db().collection(COLLECTION_FEEDBACK).document(diagnosis_id).set(doc)
    get_db().collection(COLLECTION_DIAGNOSES).document(diagnosis_id).update(
        {"has_feedback": True, "updated_at": _now()}
    )
    return doc
