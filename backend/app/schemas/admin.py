"""Admin / model-management schemas (ROUTES.md flaw #9)."""

from __future__ import annotations

from datetime import datetime

from app.schemas.common import CamelModel


class ModelVersionInfo(CamelModel):
    version: str
    model_name: str | None = None
    architecture: str | None = None
    num_classes: int | None = None
    best_epoch: int | None = None
    active: bool = False
    registered_at: datetime | None = None
    # Straight from metadata.json. test_field is the number to judge on;
    # test_studio is inflated (project_context.md caveat).
    metrics: dict = {}
    confidence_threshold: float | None = None
    temperature: float | None = None
    caveat: str | None = None


class ModelVersionList(CamelModel):
    items: list[ModelVersionInfo]
    active_version: str | None = None


class ActivateResponse(CamelModel):
    version: str
    active: bool
    detail: str


class FeedbackStats(CamelModel):
    total_diagnoses: int
    completed: int
    uncertain: int
    rejected: int
    failed: int
    feedback_count: int
    agreed: int
    corrected: int
    # Agreement rate is NOT accuracy: only farmers who bothered to respond are
    # counted, and they skew toward wrong verdicts. Labelled to prevent it
    # being read as a model metric.
    agreement_rate_of_responders: float | None = None


class AdminStats(CamelModel):
    active_model_version: str | None = None
    diagnoses: FeedbackStats
