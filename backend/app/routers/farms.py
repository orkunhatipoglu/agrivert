"""Farms & plots (ROUTES.md flaw #6).

STUB STATUS: routes, schemas and auth are wired; handlers return 501.
Firestore collections are already named in app/firebase.py
(COLLECTION_FARMS / COLLECTION_PLOTS), and diagnoses already carry farm_id /
plot_id, so implementing these is CRUD against a shape the rest of the
system already agrees on.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import CurrentUser, get_current_user, not_implemented
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

router = APIRouter(prefix="/farms", tags=["farms"])


@router.get("", response_model=FarmList, summary="List the current user's farms")
async def list_farms(user: CurrentUser = Depends(get_current_user)) -> FarmList:
    not_implemented("GET /farms")


@router.post("", response_model=Farm, status_code=status.HTTP_201_CREATED, summary="Create a farm")
async def create_farm(
    body: FarmCreate, user: CurrentUser = Depends(get_current_user)
) -> Farm:
    not_implemented("POST /farms")


@router.get("/{farm_id}", response_model=Farm, summary="Farm details")
async def get_farm(farm_id: str, user: CurrentUser = Depends(get_current_user)) -> Farm:
    not_implemented("GET /farms/{farmId}")


@router.patch("/{farm_id}", response_model=Farm, summary="Update a farm")
async def update_farm(
    farm_id: str, body: FarmUpdate, user: CurrentUser = Depends(get_current_user)
) -> Farm:
    not_implemented("PATCH /farms/{farmId}")


@router.delete(
    "/{farm_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a farm"
)
async def delete_farm(farm_id: str, user: CurrentUser = Depends(get_current_user)) -> None:
    not_implemented(
        "DELETE /farms/{farmId}",
        "Decide the cascade policy first: what happens to plots and to "
        "diagnoses that reference this farm.",
    )


@router.get(
    "/{farm_id}/plots", response_model=PlotList, summary="List plots within a farm"
)
async def list_plots(
    farm_id: str, user: CurrentUser = Depends(get_current_user)
) -> PlotList:
    not_implemented("GET /farms/{farmId}/plots")


@router.post(
    "/{farm_id}/plots",
    response_model=Plot,
    status_code=status.HTTP_201_CREATED,
    summary="Create a plot",
)
async def create_plot(
    farm_id: str, body: PlotCreate, user: CurrentUser = Depends(get_current_user)
) -> Plot:
    not_implemented("POST /farms/{farmId}/plots")


@router.patch(
    "/{farm_id}/plots/{plot_id}", response_model=Plot, summary="Update a plot"
)
async def update_plot(
    farm_id: str,
    plot_id: str,
    body: PlotUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> Plot:
    not_implemented("PATCH /farms/{farmId}/plots/{plotId}")


@router.delete(
    "/{farm_id}/plots/{plot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a plot",
)
async def delete_plot(
    farm_id: str, plot_id: str, user: CurrentUser = Depends(get_current_user)
) -> None:
    not_implemented("DELETE /farms/{farmId}/plots/{plotId}")
