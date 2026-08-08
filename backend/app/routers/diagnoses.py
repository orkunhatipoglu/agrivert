"""Diagnoses — the core photo -> verdict workflow (ROUTES.md).

Fully implemented. Modeled as an async job per flaw #4: POST validates and
enqueues, the worker does preprocessing + inference, and the client polls
GET /{id} or subscribes to GET /{id}/stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sse_starlette.sse import EventSourceResponse

from app.dependencies import CurrentUser, get_current_user, owned_or_404
from app.repositories import diagnoses as repo
from app.schemas.diagnoses import (
    TERMINAL_STATUSES,
    Diagnosis,
    DiagnosisCreated,
    DiagnosisList,
    DiagnosisListItem,
    DiagnosisStatus,
    FeedbackRequest,
    FeedbackResponse,
)
from app.services import storage
from app.services.image_validation import ImageValidationError, validate_image
from app.worker.tasks import run_diagnosis

log = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnoses", tags=["diagnoses"])

# Poll interval for the SSE stream. Firestore listeners would be pushier, but
# they need a persistent gRPC channel per connection; polling is simpler and
# adequate at the scale a job takes seconds to finish.
_STREAM_POLL_SECONDS = 1.5
_STREAM_TIMEOUT_SECONDS = 180


def _to_model(record: dict) -> Diagnosis:
    return Diagnosis(
        diagnosis_id=record["diagnosis_id"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record.get("updated_at"),
        plot_id=record.get("plot_id"),
        farm_id=record.get("farm_id"),
        crop=record.get("crop"),
        condition=record.get("condition"),
        healthy=record.get("healthy"),
        raw_label=record.get("raw_label"),
        disease_id=record.get("raw_label"),
        confidence=record.get("confidence"),
        threshold=record.get("threshold"),
        field_validated=record.get("field_validated"),
        alternatives=record.get("alternatives", []) or [],
        model_version=record.get("model_version"),
        recommendation=record.get("recommendation"),
        error=record.get("error"),
    )


@router.post(
    "",
    response_model=DiagnosisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a photo and enqueue a diagnosis",
)
async def create_diagnosis(
    file: UploadFile = File(..., description="The plant photo"),
    plot_id: str | None = Form(default=None),
    farm_id: str | None = Form(default=None),
    user: CurrentUser = Depends(get_current_user),
) -> DiagnosisCreated:
    diagnosis_id = str(uuid.uuid4())
    data = await file.read()

    # Flaw #2: validate synchronously so a bad photo fails now, with a
    # reason, instead of dying inside a worker three seconds later.
    try:
        validated = validate_image(data, file.content_type)
    except ImageValidationError as exc:
        repo.create_rejected(
            diagnosis_id=diagnosis_id,
            owner_uid=user.uid,
            reason=str(exc),
            plot_id=plot_id,
            farm_id=farm_id,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "diagnosisId": diagnosis_id,
                "status": DiagnosisStatus.REJECTED.value,
                "reason": str(exc),
            },
        )

    try:
        object_name = storage.upload_image(
            user.uid, diagnosis_id, data, validated.content_type
        )
    except Exception as exc:
        log.exception("image upload failed for %s", diagnosis_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="could not store the uploaded image; please retry",
        ) from exc

    repo.create(
        diagnosis_id=diagnosis_id,
        owner_uid=user.uid,
        image_object=object_name,
        plot_id=plot_id,
        farm_id=farm_id,
        image_meta={
            "content_type": validated.content_type,
            "width": validated.width,
            "height": validated.height,
            "size_bytes": validated.size_bytes,
        },
    )

    run_diagnosis.delay(diagnosis_id, object_name)

    return DiagnosisCreated(
        diagnosis_id=diagnosis_id, status=DiagnosisStatus.QUEUED
    )


@router.get("", response_model=DiagnosisList, summary="List diagnosis history")
async def list_diagnoses(
    farm_id: str | None = Query(default=None),
    plot_id: str | None = Query(default=None),
    disease_id: str | None = Query(
        default=None, description="Filter by raw_label, e.g. Tomato___Late_blight"
    ),
    status_filter: DiagnosisStatus | None = Query(default=None, alias="status"),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
) -> DiagnosisList:
    records = repo.list_for_user(
        owner_uid=user.uid,
        farm_id=farm_id,
        plot_id=plot_id,
        status=status_filter.value if status_filter else None,
        disease_id=disease_id,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    return DiagnosisList(
        items=[
            DiagnosisListItem(
                diagnosis_id=r["diagnosis_id"],
                status=r["status"],
                created_at=r["created_at"],
                crop=r.get("crop"),
                condition=r.get("condition"),
                healthy=r.get("healthy"),
                confidence=r.get("confidence"),
                plot_id=r.get("plot_id"),
            )
            for r in records
        ]
    )


@router.get("/{diagnosis_id}", response_model=Diagnosis, summary="Poll status/result")
async def get_diagnosis(
    diagnosis_id: str, user: CurrentUser = Depends(get_current_user)
) -> Diagnosis:
    record = owned_or_404(repo.get(diagnosis_id), user, "diagnosis")
    return _to_model(record)


@router.get("/{diagnosis_id}/stream", summary="Push status updates (SSE)")
async def stream_diagnosis(
    diagnosis_id: str, user: CurrentUser = Depends(get_current_user)
):
    """Server-sent events until the diagnosis reaches a terminal state.

    Emits only on change, so a client sees queued -> processing -> completed
    rather than a heartbeat of identical frames.
    """
    owned_or_404(repo.get(diagnosis_id), user, "diagnosis")

    async def event_source():
        last_status = None
        waited = 0.0
        while waited < _STREAM_TIMEOUT_SECONDS:
            record = repo.get(diagnosis_id)
            if record is None:
                yield {
                    "event": "error",
                    "data": json.dumps({"detail": "diagnosis disappeared"}),
                }
                return

            current = record["status"]
            if current != last_status:
                last_status = current
                payload = _to_model(record).model_dump(by_alias=True, mode="json")
                yield {"event": "status", "data": json.dumps(payload)}

            if current in {s.value for s in TERMINAL_STATUSES}:
                return

            await asyncio.sleep(_STREAM_POLL_SECONDS)
            waited += _STREAM_POLL_SECONDS

        yield {
            "event": "timeout",
            "data": json.dumps(
                {"detail": "stream timed out; fall back to polling GET /diagnoses/{id}"}
            ),
        }

    return EventSourceResponse(event_source())


@router.get("/{diagnosis_id}/image", summary="Fetch the original uploaded photo")
async def get_diagnosis_image(
    diagnosis_id: str, user: CurrentUser = Depends(get_current_user)
) -> Response:
    record = owned_or_404(repo.get(diagnosis_id), user, "diagnosis")
    object_name = record.get("image_object")
    if not object_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no image stored for this diagnosis",
        )
    try:
        data = storage.download_image(object_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="image no longer available"
        )
    content_type = (record.get("image") or {}).get("content_type", "image/jpeg")
    return Response(content=data, media_type=content_type)


@router.delete(
    "/{diagnosis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a diagnosis and its stored image",
)
async def delete_diagnosis(
    diagnosis_id: str, user: CurrentUser = Depends(get_current_user)
) -> Response:
    record = owned_or_404(repo.get(diagnosis_id), user, "diagnosis")
    if record.get("image_object"):
        storage.delete_image(record["image_object"])
    repo.delete(diagnosis_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{diagnosis_id}/feedback",
    response_model=FeedbackResponse,
    summary="Confirm or correct a verdict",
)
async def submit_feedback(
    diagnosis_id: str,
    body: FeedbackRequest,
    user: CurrentUser = Depends(get_current_user),
) -> FeedbackResponse:
    """Flaw #7. This is the retraining corpus, so it is validated strictly.

    project_context.md §3 step 6 names this loop as the real long-term fix for
    the domain gap — garbage accepted here becomes garbage training data
    later, so a correction must name a class the model actually knows.
    """
    record = owned_or_404(repo.get(diagnosis_id), user, "diagnosis")

    if record["status"] not in {
        DiagnosisStatus.COMPLETED.value,
        DiagnosisStatus.UNCERTAIN.value,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"cannot leave feedback on a diagnosis with status "
                f"{record['status']}"
            ),
        )

    corrected = body.corrected_raw_label
    if not body.agrees:
        if not corrected:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="correctedRawLabel is required when agrees is false",
            )
        if corrected != "unknown":
            from app.ml.engine import get_active_classifier

            try:
                _, clf = get_active_classifier()
                known = set(clf.classes)
            except Exception:
                known = set()
            if known and corrected not in known:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"correctedRawLabel {corrected!r} is not a known class; "
                        "use one of the model's labels or 'unknown'"
                    ),
                )

    repo.save_feedback(
        diagnosis_id=diagnosis_id,
        owner_uid=user.uid,
        agrees=body.agrees,
        corrected_raw_label=corrected,
        note=body.note,
        predicted_raw_label=record.get("raw_label"),
        model_version=record.get("model_version"),
        image_object=record.get("image_object"),
    )

    return FeedbackResponse(
        diagnosis_id=diagnosis_id,
        recorded=True,
        agrees=body.agrees,
        corrected_raw_label=corrected,
    )
