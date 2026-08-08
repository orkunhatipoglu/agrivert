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

# backend/app/config.py -> backend/ -> repo root
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_prefix: str = "/api/v1"
    environment: str = "development"

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
    # Where predict.py / data.py live. The backend imports the *training*
    # preprocessing rather than reimplementing it, so serving cannot drift
    # from training (project_context.md §2.7).
    ml_repo_root: Path = REPO_ROOT
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
