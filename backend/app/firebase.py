"""Firebase Admin SDK initialization.

Initialized lazily and exactly once per process. Both the API process and
each Celery worker process call these, so they must be import-safe and
idempotent.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import firebase_admin
from firebase_admin import credentials, firestore, storage

from app.config import get_settings

if TYPE_CHECKING:  # pragma: no cover
    from google.cloud.firestore import Client as FirestoreClient

log = logging.getLogger(__name__)

# Firestore collection names, in one place so the seeder/scripts agree.
COLLECTION_USERS = "users"
COLLECTION_FARMS = "farms"
COLLECTION_PLOTS = "plots"
COLLECTION_DIAGNOSES = "diagnoses"
COLLECTION_FEEDBACK = "diagnosis_feedback"
COLLECTION_DISEASES = "diseases"
COLLECTION_MODEL_VERSIONS = "model_versions"


@lru_cache
def get_app() -> firebase_admin.App:
    """Initialize (once) and return the default Firebase app."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass  # not yet initialized

    settings = get_settings()
    options: dict[str, str] = {}
    if settings.firebase_storage_bucket:
        options["storageBucket"] = settings.firebase_storage_bucket
    if settings.firebase_project_id:
        options["projectId"] = settings.firebase_project_id

    if settings.firebase_credentials_path:
        cred = credentials.Certificate(settings.firebase_credentials_path)
    else:
        # Application Default Credentials: GOOGLE_APPLICATION_CREDENTIALS,
        # gcloud login, or workload identity on GCP.
        cred = credentials.ApplicationDefault()

    log.info("initializing firebase app (project=%s)", settings.firebase_project_id)
    return firebase_admin.initialize_app(cred, options or None)


@lru_cache
def get_db() -> "FirestoreClient":
    return firestore.client(app=get_app())


@lru_cache
def get_bucket():
    settings = get_settings()
    if not settings.firebase_storage_bucket:
        raise RuntimeError(
            "FIREBASE_STORAGE_BUCKET is not configured; image upload/download "
            "cannot work without it."
        )
    return storage.bucket(app=get_app())
