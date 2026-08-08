"""Inference engine wrapper.

Wraps `predict.DiseaseClassifier` from the ML repo rather than
reimplementing preprocessing. That is deliberate: `predict.py` builds its
eval transform from `data.build_eval_transform`, so the backend applies
byte-identical preprocessing to what training used. Reimplementing
resize/crop/normalize here would reintroduce exactly the train/serve drift
project_context.md §2.7 warns about.

The classifier is cached per version and loaded lazily, so a Celery worker
pays the ~26MB checkpoint load once, on its first task.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.ml.registry import ModelVersion, resolve_active_version

log = logging.getLogger(__name__)

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _ensure_ml_repo_importable() -> None:
    """Put the ML repo root on sys.path so `predict` and `data` import."""
    root = str(Path(get_settings().ml_repo_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_classifier(version: ModelVersion):
    """Return a cached DiseaseClassifier for this version.

    Thread-safe and idempotent: concurrent first-calls load once.
    """
    cached = _cache.get(version.version)
    if cached is not None:
        return cached

    with _lock:
        cached = _cache.get(version.version)
        if cached is not None:
            return cached

        _ensure_ml_repo_importable()
        try:
            from predict import DiseaseClassifier
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "could not import predict.DiseaseClassifier from "
                f"{get_settings().ml_repo_root}. Is ML_REPO_ROOT correct and are "
                "torch/albumentations installed?"
            ) from exc

        log.info("loading model version %s from %s", version.version, version.path)
        clf = DiseaseClassifier(
            version.path, device=get_settings().inference_device
        )
        _cache[version.version] = clf
        return clf


def get_active_classifier() -> tuple[str, Any]:
    """(version_name, classifier) for the currently active version."""
    version = resolve_active_version()
    return version.version, load_classifier(version)


def clear_cache() -> None:
    """Drop cached classifiers.

    Called after an activation so the next task picks the new version up.
    Note this only clears *this* process — see app/routers/admin.py on why
    activation is not instantaneous across a worker pool.
    """
    with _lock:
        _cache.clear()


def predict_image(image_bytes: bytes, top_k: int = 3) -> dict:
    """Run inference and return predict.py's verdict dict, plus the version.

    The returned `status` is already "completed" or "uncertain" — the
    threshold decision lives in predict.py and is NOT re-litigated here.
    """
    version_name, clf = get_active_classifier()
    result = clf.predict(image_bytes, top_k=top_k)
    # predict.py reports metadata's model_name; the API contract wants the
    # registry version that produced it (ROUTES.md inference notes).
    result["model_version"] = version_name
    result["model_name"] = clf.metadata.get("model_name")
    return result
