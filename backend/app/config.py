"""Runtime configuration.

Everything environment-specific lives here. Nothing about the *model* lives
here beyond where to find the registry — model parameters (normalization,
crop size, temperature, threshold) are read from each version's
metadata.json at load time, per project_context.md §2.7 ("Serve from
metadata.json, never hardcode").
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_prefix: str = "/api/v1"
    environment: str = "development"

    # --- CORS -----------------------------------------------------------
    # Comma-separated origins allowed to call the API from a browser. The
    # default covers the Next.js dev server only; a deployment must set its
    # real origin. Credentials are sent on these requests, so "*" is not a
    # safe default and is rejected outright below.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if "*" in origins:
            raise ValueError(
                "CORS_ORIGINS cannot be '*': the API sends credentialed "
                "requests, and browsers reject a wildcard with credentials. "
                "List the frontend's real origin(s) instead."
            )
        return origins

    # --- Firebase -------------------------------------------------------
    # Path to a service-account JSON, or leave unset to use
    # GOOGLE_APPLICATION_CREDENTIALS / workload identity.
    firebase_credentials_path: str | None = None
    firebase_project_id: str | None = None
    firebase_storage_bucket: str | None = None

    # --- Celery / Redis -------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_task_time_limit: int = 120
    celery_task_soft_time_limit: int = 100

    # --- Model registry -------------------------------------------------
    # Directory holding versioned model dirs:
    #   models/<version>/{best.pt,metadata.json,labels.json}
    model_registry_dir: Path = BACKEND_DIR / "models"
    # Fallback when Firestore has no active version recorded (local dev).
    default_model_version: str | None = None
    # Repo root, which must contain the `ml/` package. The backend imports
    # ml.predict (which imports ml.data) so serving preprocessing IS the
    # training preprocessing — the same objects, not a copy that can drift.
    ml_repo_root: Path = BACKEND_DIR.parent
    inference_device: str | None = None  # None => cuda if available, else cpu

    # --- Upload validation (ROUTES.md flaw #2) --------------------------
    max_upload_bytes: int = 12 * 1024 * 1024
    min_image_dimension: int = 64
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    # --- Storage --------------------------------------------------------
    # Object-name prefix inside the Firebase Storage bucket.
    storage_image_prefix: str = "diagnoses"


@lru_cache
def get_settings() -> Settings:
    return Settings()
