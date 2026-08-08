"""Diagnosis request/response schemas.

Field names are camelCase on the wire (ROUTES.md uses `diagnosisId`,
`modelVersion`) while staying snake_case in Python.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, from_attributes=True
    )


class DiagnosisStatus(str, Enum):
    """The exact status set ROUTES.md specifies for GET /diagnoses/{id}."""

    QUEUED = "queued"
    PROCESSING = "processing"
    REJECTED = "rejected"      # upload failed validation (flaw #2)
    UNCERTAIN = "uncertain"    # below confidence threshold (§2.7)
    COMPLETED = "completed"
    FAILED = "failed"          # inference errored


TERMINAL_STATUSES = frozenset(
    {
        DiagnosisStatus.REJECTED,
        DiagnosisStatus.UNCERTAIN,
        DiagnosisStatus.COMPLETED,
        DiagnosisStatus.FAILED,
    }
)


class Alternative(CamelModel):
    raw_label: str
    crop: str
    condition: str
    confidence: float


class DiagnosisCreated(CamelModel):
    diagnosis_id: str
    status: DiagnosisStatus


class Diagnosis(CamelModel):
    diagnosis_id: str
    status: DiagnosisStatus
    created_at: datetime
    updated_at: datetime | None = None
    plot_id: str | None = None
    farm_id: str | None = None

    # Populated once completed. All None while queued/processing, and also
    # when uncertain — an uncertain verdict must NOT surface a diagnosis
    # (project_context.md §2.7: a confident wrong answer is worse than an
    # honest "can't tell").
    crop: str | None = None
    condition: str | None = None
    healthy: bool | None = None
    raw_label: str | None = None
    disease_id: str | None = None

    confidence: float | None = None
    threshold: float | None = None
    # False => the model never saw a real field photo of this class.
    field_validated: bool | None = None
    alternatives: list[Alternative] = Field(default_factory=list)

    # Always present once completed, per ROUTES.md's inference notes.
    model_version: str | None = None
    recommendation: str | None = None

    # Set when status is rejected or failed.
    error: str | None = None


class DiagnosisListItem(CamelModel):
    diagnosis_id: str
    status: DiagnosisStatus
    created_at: datetime
    crop: str | None = None
    condition: str | None = None
    healthy: bool | None = None
    confidence: float | None = None
    plot_id: str | None = None


class DiagnosisList(CamelModel):
    items: list[DiagnosisListItem]
    next_page_token: str | None = None


class FeedbackRequest(CamelModel):
    """Farmer confirms or corrects a verdict (ROUTES.md flaw #7).

    This is the long-term fix for the domain gap: project_context.md §3 step 6
    names this loop as the real source of field training data.
    """

    agrees: bool
    corrected_raw_label: str | None = Field(
        default=None,
        description=(
            "Required when agrees is false. Must be one of the model's known "
            "classes, or 'unknown' if the farmer cannot identify it."
        ),
    )
    note: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(CamelModel):
    diagnosis_id: str
    recorded: bool
    agrees: bool
    corrected_raw_label: str | None = None
