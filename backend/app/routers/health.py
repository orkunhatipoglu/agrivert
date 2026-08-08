"""Liveness/readiness."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.common import HealthResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["misc"])


@router.get("/health", response_model=HealthResponse, summary="Liveness/readiness check")
async def health() -> HealthResponse:
    """Report dependency readiness, not just process liveness.

    Resolves the active model version *without* loading weights — a 26MB
    checkpoint load on every health probe would be its own outage.
    """
    settings = get_settings()
    problems: list[str] = []

    model_version = None
    model_ready = False
    try:
        from app.ml.registry import resolve_active_version

        model_version = resolve_active_version().version
        model_ready = True
    except Exception as exc:
        problems.append(f"model: {exc}")

    firestore_ready = False
    try:
        from app.firebase import get_db

        next(get_db().collections(), None)
        firestore_ready = True
    except Exception as exc:
        problems.append(f"firestore: {exc}")

    broker_ready = False
    try:
        from redis import Redis

        Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        broker_ready = True
    except Exception as exc:
        problems.append(f"broker: {exc}")

    ok = model_ready and firestore_ready and broker_ready
    return HealthResponse(
        status="ok" if ok else "degraded",
        environment=settings.environment,
        model_version=model_version,
        model_ready=model_ready,
        firestore_ready=firestore_ready,
        broker_ready=broker_ready,
        detail="; ".join(problems) or None,
    )
