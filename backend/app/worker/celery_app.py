"""Celery application.

Run with:
    celery -A app.worker.celery_app:celery_app worker --loglevel=info --concurrency=1

Concurrency note: each worker *process* loads its own ~26MB checkpoint into
its own CUDA context. On a single RTX 4060, prefer concurrency=1 with more
replicas over high concurrency in one process.
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "agrivert",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    worker_prefetch_multiplier=1,
    # A retried inference is cheap; a lost diagnosis is not.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
