"""Farms HTTP contract tests.

These guard the thing that actually broke frontend/backend exchange: the wire
format. `lib/api.ts` sends camelCase (`cropType`, `areaHectares`) and reads
camelCase (`farmId`, `plotId`) back, while the Python side is snake_case.
The CamelModel alias config is what bridges them, and it is easy to break
without noticing — nothing else fails loudly when it does.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import CurrentUser, get_current_user
from tests.test_farms import FakeDb  # reuse the in-memory Firestore double


@pytest.fixture
def client(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr("app.repositories.farms.get_db", lambda: db)

    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        uid="user-1", email="a@example.com", claims={}
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_farm_roundtrip_is_camel_case(client):
    created = client.post("/api/v1/farms", json={"name": "Ana Tarla", "region": "Aydın"})
    assert created.status_code == 201, created.text

    body = created.json()
    # Exactly the keys lib/api.ts destructures.
    assert body["name"] == "Ana Tarla"
    assert body["region"] == "Aydın"
    assert "farmId" in body and body["farmId"]
    assert "ownerUid" in body
    assert "createdAt" in body
    # snake_case must NOT leak through.
    assert "farm_id" not in body

    listed = client.get("/api/v1/farms")
    assert listed.status_code == 200
    assert [f["farmId"] for f in listed.json()["items"]] == [body["farmId"]]


def test_plot_accepts_and_returns_camel_case(client):
    farm_id = client.post("/api/v1/farms", json={"name": "F"}).json()["farmId"]

    created = client.post(
        f"/api/v1/farms/{farm_id}/plots",
        json={"name": "North", "cropType": "Tomato", "areaHectares": 2.5},
    )
    assert created.status_code == 201, created.text

    plot = created.json()
    assert plot["cropType"] == "Tomato"
    assert plot["areaHectares"] == 2.5
    assert plot["farmId"] == farm_id
    assert "crop_type" not in plot


def test_patch_farm_updates_only_given_fields(client):
    farm_id = client.post(
        "/api/v1/farms", json={"name": "Old", "region": "Aydın"}
    ).json()["farmId"]

    patched = client.patch(f"/api/v1/farms/{farm_id}", json={"name": "New"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "New"
    assert patched.json()["region"] == "Aydın"


def test_plot_from_another_farm_is_404(client):
    """`/farms/{a}/plots/{p}` must not reach a plot that lives in farm b."""
    farm_a = client.post("/api/v1/farms", json={"name": "A"}).json()["farmId"]
    farm_b = client.post("/api/v1/farms", json={"name": "B"}).json()["farmId"]
    plot_b = client.post(
        f"/api/v1/farms/{farm_b}/plots",
        json={"name": "B1", "cropType": "Corn"},
    ).json()["plotId"]

    resp = client.patch(
        f"/api/v1/farms/{farm_a}/plots/{plot_b}", json={"name": "hijack"}
    )
    assert resp.status_code == 404


def test_delete_farm_returns_204_and_cascades(client):
    farm_id = client.post("/api/v1/farms", json={"name": "F"}).json()["farmId"]
    client.post(
        f"/api/v1/farms/{farm_id}/plots", json={"name": "P", "cropType": "Tomato"}
    )

    assert client.delete(f"/api/v1/farms/{farm_id}").status_code == 204
    assert client.get(f"/api/v1/farms/{farm_id}").status_code == 404


def test_area_must_be_positive(client):
    """Mirrors the client-side guard in the plot dialog."""
    farm_id = client.post("/api/v1/farms", json={"name": "F"}).json()["farmId"]
    resp = client.post(
        f"/api/v1/farms/{farm_id}/plots",
        json={"name": "P", "cropType": "Tomato", "areaHectares": 0},
    )
    assert resp.status_code == 422


def test_notifications_routes_are_gone(client):
    assert client.get("/api/v1/notifications").status_code == 404
