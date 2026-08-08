"""Firestore access for farms and plots (ROUTES.md flaw #6).

Farms and plots live in two top-level collections rather than plots being a
subcollection of farms. Diagnoses reference a `plot_id` directly and history
filters on it without knowing the farm, so a top-level collection keeps that
a single lookup instead of a collection-group query.

Both documents carry `owner_uid`, so every read is ownership-checkable
without walking up to the parent farm.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore as gcf

from app.firebase import COLLECTION_FARMS, COLLECTION_PLOTS, get_db

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None.

    PATCH bodies are partial: an omitted field arrives as None and must leave
    the stored value alone, not overwrite it with null.
    """
    return {k: v for k, v in values.items() if v is not None}


# --- Farms ----------------------------------------------------------------


def create_farm(owner_uid: str, name: str, region: str | None, location: dict | None) -> dict:
    farm_id = str(uuid.uuid4())
    doc = {
        "farm_id": farm_id,
        "owner_uid": owner_uid,
        "name": name,
        "region": region,
        "location": location,
        "created_at": _now(),
        "updated_at": _now(),
    }
    get_db().collection(COLLECTION_FARMS).document(farm_id).set(doc)
    return doc


def get_farm(farm_id: str) -> dict | None:
    snap = get_db().collection(COLLECTION_FARMS).document(farm_id).get()
    return snap.to_dict() if snap.exists else None


def list_farms(owner_uid: str) -> list[dict]:
    q = get_db().collection(COLLECTION_FARMS).where(
        filter=gcf.FieldFilter("owner_uid", "==", owner_uid)
    )
    farms = [snap.to_dict() for snap in q.stream()]
    # Sorted in Python rather than with order_by: combining a where and an
    # order_by on different fields needs a composite index, and this list is
    # small enough per user that it isn't worth requiring one.
    farms.sort(key=lambda f: f.get("name") or "")
    return farms


def update_farm(farm_id: str, changes: dict) -> dict | None:
    payload = _clean(changes)
    if not payload:
        return get_farm(farm_id)
    payload["updated_at"] = _now()
    get_db().collection(COLLECTION_FARMS).document(farm_id).update(payload)
    return get_farm(farm_id)


def delete_farm(farm_id: str) -> int:
    """Delete a farm and every plot inside it. Returns the plot count.

    Cascading to plots is deliberate: leaving them behind would strand rows
    that reference a farm that no longer exists, and nothing else can reach
    them. Diagnoses are NOT touched — a diagnosis is a historical record of
    something that was actually observed, so it outlives the plot it was
    filed against and keeps its now-dangling plot_id.
    """
    db = get_db()
    plots = list(
        db.collection(COLLECTION_PLOTS)
        .where(filter=gcf.FieldFilter("farm_id", "==", farm_id))
        .stream()
    )
    batch = db.batch()
    for i, snap in enumerate(plots):
        batch.delete(snap.reference)
        if (i + 1) % 400 == 0:  # Firestore caps a batch at 500 writes
            batch.commit()
            batch = db.batch()
    batch.delete(db.collection(COLLECTION_FARMS).document(farm_id))
    batch.commit()
    log.info("deleted farm %s and %d plot(s)", farm_id, len(plots))
    return len(plots)


# --- Plots ----------------------------------------------------------------


def create_plot(
    owner_uid: str,
    farm_id: str,
    name: str,
    crop_type: str,
    area_hectares: float | None,
    location: dict | None,
) -> dict:
    plot_id = str(uuid.uuid4())
    doc = {
        "plot_id": plot_id,
        "farm_id": farm_id,
        "owner_uid": owner_uid,
        "name": name,
        "crop_type": crop_type,
        "area_hectares": area_hectares,
        "location": location,
        "created_at": _now(),
        "updated_at": _now(),
    }
    get_db().collection(COLLECTION_PLOTS).document(plot_id).set(doc)
    return doc


def get_plot(plot_id: str) -> dict | None:
    snap = get_db().collection(COLLECTION_PLOTS).document(plot_id).get()
    return snap.to_dict() if snap.exists else None


def list_plots(farm_id: str) -> list[dict]:
    q = get_db().collection(COLLECTION_PLOTS).where(
        filter=gcf.FieldFilter("farm_id", "==", farm_id)
    )
    plots = [snap.to_dict() for snap in q.stream()]
    plots.sort(key=lambda p: p.get("name") or "")
    return plots


def update_plot(plot_id: str, changes: dict) -> dict | None:
    payload = _clean(changes)
    if not payload:
        return get_plot(plot_id)
    payload["updated_at"] = _now()
    get_db().collection(COLLECTION_PLOTS).document(plot_id).update(payload)
    return get_plot(plot_id)


def delete_plot(plot_id: str) -> None:
    get_db().collection(COLLECTION_PLOTS).document(plot_id).delete()
