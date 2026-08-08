"""Firestore access for the disease knowledge base.

Documents are keyed by `raw_label` (e.g. `Tomato___Late_blight`), which is
also what a diagnosis carries, so a verdict maps to its KB entry without a
lookup table.
"""

from __future__ import annotations

import logging

from app.firebase import COLLECTION_DISEASES, get_db

log = logging.getLogger(__name__)


def get(disease_id: str) -> dict | None:
    snap = get_db().collection(COLLECTION_DISEASES).document(disease_id).get()
    return snap.to_dict() if snap.exists else None


def list_all() -> list[dict]:
    return [snap.to_dict() for snap in get_db().collection(COLLECTION_DISEASES).stream()]


def recommendation_for(raw_label: str | None) -> str | None:
    """Treatment guidance to attach to a completed diagnosis.

    Returns None unless a human has filled in the treatment steps AND marked
    the entry reviewed. That gate is the point: `seed_diseases.py` creates
    every class with empty content and `content_reviewed: false`, and
    unreviewed text must never reach a farmer as advice about what to put on
    a real crop.
    """
    if not raw_label:
        return None

    try:
        doc = get(raw_label)
    except Exception as exc:  # KB is supplementary; never fail a diagnosis
        log.warning("could not read disease %s for recommendation: %s", raw_label, exc)
        return None

    if not doc or not doc.get("content_reviewed"):
        return None

    steps = [s for s in (doc.get("treatment") or []) if s]
    if not steps:
        return None
    return " ".join(steps) if len(steps) == 1 else "\n".join(f"- {s}" for s in steps)
