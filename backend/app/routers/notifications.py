"""Notifications (ROUTES.md marks these optional, v1.1+).

STUB STATUS: not implemented. Regional outbreak alerts depend on farm
location (flaw #6) and on enough diagnosis volume per region to make
"trending" mean anything, so this is correctly last in line.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import CurrentUser, get_current_user, not_implemented
from app.schemas.common import CamelModel

router = APIRouter(prefix="/notifications", tags=["notifications"])


class Notification(CamelModel):
    notification_id: str
    kind: str
    title: str
    body: str | None = None
    read: bool = False


class NotificationList(CamelModel):
    items: list[Notification]


class SubscribeRequest(CamelModel):
    region: str | None = None
    fcm_token: str | None = None


@router.get("", response_model=NotificationList, summary="List alerts")
async def list_notifications(
    user: CurrentUser = Depends(get_current_user),
) -> NotificationList:
    not_implemented("GET /notifications")


@router.post(
    "/subscribe",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Opt into regional outbreak alerts",
)
async def subscribe(
    body: SubscribeRequest, user: CurrentUser = Depends(get_current_user)
) -> None:
    not_implemented(
        "POST /notifications/subscribe",
        "Firebase Cloud Messaging is the natural transport here since the "
        "project is already on Firebase.",
    )
