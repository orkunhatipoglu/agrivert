"""Farm and plot schemas (ROUTES.md flaw #6 — scoping enables history/trends)."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import CamelModel


class GeoPoint(CamelModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class FarmCreate(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    location: GeoPoint | None = None
    region: str | None = Field(default=None, max_length=200)


class FarmUpdate(CamelModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    location: GeoPoint | None = None
    region: str | None = Field(default=None, max_length=200)


class Farm(CamelModel):
    farm_id: str
    owner_uid: str
    name: str
    location: GeoPoint | None = None
    region: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class PlotCreate(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    crop_type: str = Field(min_length=1, max_length=120)
    area_hectares: float | None = Field(default=None, gt=0)
    location: GeoPoint | None = None


class PlotUpdate(CamelModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    crop_type: str | None = Field(default=None, min_length=1, max_length=120)
    area_hectares: float | None = Field(default=None, gt=0)
    location: GeoPoint | None = None


class Plot(CamelModel):
    plot_id: str
    farm_id: str
    name: str
    crop_type: str
    area_hectares: float | None = None
    location: GeoPoint | None = None
    created_at: datetime
    updated_at: datetime | None = None


class FarmList(CamelModel):
    items: list[Farm]


class PlotList(CamelModel):
    items: list[Plot]
