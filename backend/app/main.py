"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import admin, auth, diagnoses, diseases, farms, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve the active model at startup — report, don't crash.

    The API process does not run inference (the Celery workers do), so a
    missing model must not stop the API from serving history and health.
    It is still logged loudly, because a missing model means new diagnoses
    will fail.
    """
    try:
        from app.ml.registry import resolve_active_version

        version = resolve_active_version()
        log.info("active model version: %s (%s)", version.version, version.path)
    except Exception as exc:
        log.error("NO USABLE MODEL VERSION: %s", exc)
    yield


app = FastAPI(
    title="Agrivert API",
    version="0.1.0",
    description=(
        "Plant disease diagnosis API. Routes follow ROUTES.md; the diagnoses "
        "pipeline is implemented, other route groups are scaffolded stubs "
        "returning 501."
    ),
    lifespan=lifespan,
)

# Origins come from CORS_ORIGINS and default to the Next.js dev server.
# A wildcard is rejected in config.py rather than accepted here: these are
# credentialed requests, so "*" would not work in a browser anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

prefix = settings.api_prefix
app.include_router(auth.router, prefix=prefix)
app.include_router(farms.router, prefix=prefix)
app.include_router(diagnoses.router, prefix=prefix)
app.include_router(diseases.router, prefix=prefix)
app.include_router(admin.router, prefix=prefix)
app.include_router(health.router, prefix=prefix)
