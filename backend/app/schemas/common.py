"""Shared schema pieces."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, from_attributes=True
    )


class ErrorDetail(CamelModel):
    detail: str


class HealthResponse(CamelModel):
    status: str
    environment: str
    # Readiness is more than "the process is up": if no model version
    # resolves, the diagnoses pipeline cannot serve, so /health reports it.
    model_version: str | None = None
    model_ready: bool = False
    firestore_ready: bool = False
    broker_ready: bool = False
    detail: str | None = None
