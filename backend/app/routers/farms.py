"""Farms & plots (ROUTES.md flaw #6).

Scoping a diagnosis to a plot is what turns single verdicts into a trend, so
this is CRUD in service of the history filters on `GET /diagnoses`.

Ownership is enforced on every route via `owned_or_404`: both farm and plot
documents carry `owner_uid`, and a plot is additionally checked against the
farm in its path so `/farms/{a}/plots/{p}` cannot reach a plot belonging to
farm `b`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.dependencies import CurrentUser, get_current_user, owned_or_404
from app.repositories import farms as repo
from app.schemas.farms import (
    Farm,
    FarmCreate,
    FarmList,
    FarmUpdate,
    Plot,
    PlotCreate,
    PlotList,
    PlotUpdate,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/farms", tags=["farms"])


def _require_farm(farm_id: str, user: CurrentUser) -> dict:
    return owned_or_404(repo.get_farm(farm_id), user, "farm")


def _require_plot(farm_id: str, plot_id: str, user: CurrentUser) -> dict:
    """Resolve a plot, checking both ownership and that it is in this farm."""
    _require_farm(farm_id, user)
    plot = owned_or_404(repo.get_plot(plot_id), user, "plot")
    if plot.get("farm_id") != farm_id:
        # 404 rather than 400: the plot exists but not at this path, and
        # saying so would confirm an id the caller shouldn't be probing.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="plot not found"
        )
    return plot


@router.get("", response_model=FarmList, summary="List the current user's farms")
async def list_farms(user: CurrentUser = Depends(get_current_user)) -> FarmList:
    return FarmList(items=[Farm(**f) for f in repo.list_farms(user.uid)])


@router.post(
    "", response_model=Farm, status_code=status.HTTP_201_CREATED, summary="Create a farm"
)
async def create_farm(
    body: FarmCreate, user: CurrentUser = Depends(get_current_user)
) -> Farm:
    doc = repo.create_farm(
        owner_uid=user.uid,
        name=body.name,
        region=body.region,
        location=body.location.model_dump() if body.location else None,
    )
    return Farm(**doc)


@router.get("/{farm_id}", response_model=Farm, summary="Farm details")
async def get_farm(farm_id: str, user: CurrentUser = Depends(get_current_user)) -> Farm:
    return Farm(**_require_farm(farm_id, user))


@router.patch("/{farm_id}", response_model=Farm, summary="Update a farm")
async def update_farm(
    farm_id: str, body: FarmUpdate, user: CurrentUser = Depends(get_current_user)
) -> Farm:
    _require_farm(farm_id, user)
    updated = repo.update_farm(
        farm_id,
        {
            "name": body.name,
            "region": body.region,
            "location": body.location.model_dump() if body.location else None,
        },
    )
    return Farm(**updated)


@router.delete(
    "/{farm_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a farm"
)
async def delete_farm(
    farm_id: str, user: CurrentUser = Depends(get_current_user)
) -> Response:
    """Deletes the farm and cascades to its plots.

    Diagnoses are left alone — they record something that was actually
    observed in a field, so they outlive the plot they were filed against.
    """
    _require_farm(farm_id, user)
    removed = repo.delete_farm(farm_id)
    log.info("user %s deleted farm %s (%d plots)", user.uid, farm_id, removed)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{farm_id}/plots", response_model=PlotList, summary="List plots within a farm"
)
async def list_plots(
    farm_id: str, user: CurrentUser = Depends(get_current_user)
) -> PlotList:
    _require_farm(farm_id, user)
    return PlotList(items=[Plot(**p) for p in repo.list_plots(farm_id)])


@router.post(
    "/{farm_id}/plots",
    response_model=Plot,
    status_code=status.HTTP_201_CREATED,
    summary="Create a plot",
)
async def create_plot(
    farm_id: str, body: PlotCreate, user: CurrentUser = Depends(get_current_user)
) -> Plot:
    _require_farm(farm_id, user)
    doc = repo.create_plot(
        owner_uid=user.uid,
        farm_id=farm_id,
        name=body.name,
        crop_type=body.crop_type,
        area_hectares=body.area_hectares,
        location=body.location.model_dump() if body.location else None,
    )
    return Plot(**doc)


@router.patch(
    "/{farm_id}/plots/{plot_id}", response_model=Plot, summary="Update a plot"
)
async def update_plot(
    farm_id: str,
    plot_id: str,
    body: PlotUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> Plot:
    _require_plot(farm_id, plot_id, user)
    updated = repo.update_plot(
        plot_id,
        {
            "name": body.name,
            "crop_type": body.crop_type,
            "area_hectares": body.area_hectares,
            "location": body.location.model_dump() if body.location else None,
        },
    )
    return Plot(**updated)


@router.delete(
    "/{farm_id}/plots/{plot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a plot",
)
async def delete_plot(
    farm_id: str, plot_id: str, user: CurrentUser = Depends(get_current_user)
) -> Response:
    _require_plot(farm_id, plot_id, user)
    repo.delete_plot(plot_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
