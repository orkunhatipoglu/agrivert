"""Admin / model management (ROUTES.md flaw #9).

The model-version routes are implemented, since they are the mechanism that
makes swapping a retrained model a config change. /admin/stats is stubbed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import CurrentUser, not_implemented, require_admin
from app.ml import registry
from app.ml.registry import ModelRegistryError
from app.repositories import models as model_repo
from app.schemas.admin import ActivateResponse, AdminStats, ModelVersionInfo, ModelVersionList

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/models",
    response_model=ModelVersionList,
    summary="List registered model versions and their eval metrics",
)
async def list_models(_: CurrentUser = Depends(require_admin)) -> ModelVersionList:
    """Join what's on disk with what Firestore knows.

    Disk is the source of truth for *existence* — a Firestore record whose
    directory is gone is not listed as available, because activating it would
    take the pipeline down.
    """
    on_disk = {v.version: v for v in registry.discover_versions()}
    records = {r["version"]: r for r in model_repo.list_records()}

    active_version = None
    items = []
    for version, mv in sorted(on_disk.items()):
        record = records.get(version, {})
        if record.get("active"):
            active_version = version
        summary = mv.summary()
        items.append(
            ModelVersionInfo(
                version=version,
                model_name=summary.get("model_name"),
                architecture=summary.get("architecture"),
                num_classes=summary.get("num_classes"),
                best_epoch=summary.get("best_epoch"),
                active=bool(record.get("active", False)),
                registered_at=record.get("registered_at"),
                metrics=summary.get("metrics", {}),
                confidence_threshold=summary.get("confidence_threshold"),
                temperature=summary.get("temperature"),
                caveat=summary.get("caveat"),
            )
        )

    orphaned = set(records) - set(on_disk)
    if orphaned:
        log.warning(
            "model versions registered in Firestore but missing on disk: %s",
            ", ".join(sorted(orphaned)),
        )

    return ModelVersionList(items=items, active_version=active_version)


@router.post(
    "/models/{version}/activate",
    response_model=ActivateResponse,
    summary="Promote a model version to production",
)
async def activate_model(
    version: str, _: CurrentUser = Depends(require_admin)
) -> ActivateResponse:
    """Point serving at `version`.

    Validated against disk first: activating a version whose weights are
    missing would leave every worker unable to resolve a model.
    """
    try:
        resolved = registry.get_version(version)
    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    model_repo.register(version, metrics=resolved.summary().get("metrics", {}))
    model_repo.activate(version)

    # Clear this process's cache so anything served from the API side
    # reloads. Workers hold their own caches, but they re-resolve the active
    # version per task, so they pick this up on their next job.
    from app.ml.engine import clear_cache

    clear_cache()

    return ActivateResponse(
        version=version,
        active=True,
        detail=(
            "Activated. Workers re-resolve the active version on each task, "
            "so the next diagnosis uses this version — no restart needed. "
            "A job already in flight finishes on the previous version."
        ),
    )


@router.get(
    "/stats",
    response_model=AdminStats,
    summary="Aggregate accuracy/feedback stats across diagnoses",
)
async def admin_stats(_: CurrentUser = Depends(require_admin)) -> AdminStats:
    not_implemented(
        "GET /admin/stats",
        "Needs aggregation over the diagnoses and diagnosis_feedback "
        "collections; Firestore has no GROUP BY, so this wants either "
        "incrementing counters on write or a scheduled rollup job.",
    )
