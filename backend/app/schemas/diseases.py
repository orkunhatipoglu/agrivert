"""Disease knowledge-base schemas (ROUTES.md flaw #5).

The *shape* is defined here and seeded by scripts/seed_diseases.py from the
model's own labels.json, so the KB can never drift out of sync with what the
model can actually predict. The *content* (description, symptoms, treatment)
is intentionally left empty — see scripts/seed_diseases.py.
"""

from __future__ import annotations

from enum import Enum

from app.schemas.common import CamelModel


class Severity(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DiseaseSummary(CamelModel):
    disease_id: str
    raw_label: str
    crop: str
    condition: str
    healthy: bool
    # False => model has never seen a real field photo of this class
    # (project_context.md §2.3). Surfaced so the frontend can hedge.
    field_validated: bool


class Disease(DiseaseSummary):
    description: str | None = None
    symptoms: list[str] = []
    treatment: list[str] = []
    prevention: list[str] = []
    severity: Severity = Severity.UNKNOWN
    references: list[str] = []
    # True once a human has filled in and reviewed the agronomic content.
    content_reviewed: bool = False


class DiseaseList(CamelModel):
    items: list[DiseaseSummary]
