"""Celery tasks.

The image is passed by *reference* (storage object name), not by value.
Shipping 12MB of JPEG through Redis for every diagnosis would make the broker
the bottleneck and leave photo bytes sitting in the queue backlog.
"""

from __future__ import annotations

import logging

from celery.signals import worker_process_init
from celery.utils.log import get_task_logger

from app.repositories import diagnoses as repo
from app.services import storage
from app.worker.celery_app import celery_app

log = get_task_logger(__name__)


@worker_process_init.connect
def _warm_model(**_kwargs) -> None:
    """Load the model when the worker process starts, not on first request.

    Without this the first farmer to submit a photo pays the checkpoint-load
    latency. Failure here is logged, not raised: a worker that can't preload
    should still start and report per-task errors rather than crash-looping.
    """
    try:
        from app.ml.engine import get_active_classifier

        version, _ = get_active_classifier()
        log.info("worker warmed up with model version %s", version)
    except Exception as exc:
        log.warning("model preload failed (will retry per-task): %s", exc)


@celery_app.task(
    name="agrivert.run_diagnosis",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def run_diagnosis(self, diagnosis_id: str, image_object: str) -> dict:
    """Fetch the image, run inference, persist the verdict."""
    log.info("running diagnosis %s", diagnosis_id)

    try:
        repo.mark_processing(diagnosis_id)
    except Exception as exc:
        log.error("could not mark %s processing: %s", diagnosis_id, exc)

    try:
        image_bytes = storage.download_image(image_object)
    except FileNotFoundError as exc:
        # The object is gone; retrying will not conjure it back.
        log.error("image missing for %s: %s", diagnosis_id, exc)
        repo.mark_failed(diagnosis_id, "uploaded image could not be retrieved")
        return {"diagnosis_id": diagnosis_id, "status": "failed"}
    except Exception as exc:
        log.warning("storage error for %s: %s", diagnosis_id, exc)
        raise self.retry(exc=exc)

    try:
        from app.ml.engine import predict_image

        result = predict_image(image_bytes)
    except Exception as exc:
        log.exception("inference failed for %s", diagnosis_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        repo.mark_failed(diagnosis_id, f"inference failed: {exc}")
        return {"diagnosis_id": diagnosis_id, "status": "failed"}

    try:
        repo.save_result(diagnosis_id, result)
    except Exception as exc:
        log.exception("could not persist result for %s", diagnosis_id)
        raise self.retry(exc=exc)

    log.info(
        "diagnosis %s -> %s (confidence=%.4f, model=%s)",
        diagnosis_id,
        result["status"],
        result.get("confidence", float("nan")),
        result.get("model_version"),
    )
    return {
        "diagnosis_id": diagnosis_id,
        "status": result["status"],
        "model_version": result.get("model_version"),
    }
