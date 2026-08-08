"""Farms & plots tests.

Firestore is faked with an in-memory double rather than mocked call-by-call,
so these exercise the real repository logic (partial updates, cascade delete,
cross-farm plot access) instead of asserting that a mock was called.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.dependencies import CurrentUser, owned_or_404


class FakeSnapshot:
    def __init__(self, doc_id, data, reference=None):
        self.id = doc_id
        self._data = data
        self.reference = reference

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class FakeDocument:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self.id = doc_id

    def set(self, data):
        self._collection.docs[self.id] = dict(data)

    def get(self):
        return FakeSnapshot(self.id, self._collection.docs.get(self.id), self)

    def update(self, data):
        if self.id not in self._collection.docs:
            raise KeyError(self.id)
        self._collection.docs[self.id].update(data)

    def delete(self):
        self._collection.docs.pop(self.id, None)


class FakeQuery:
    def __init__(self, collection, filters=None):
        self._collection = collection
        self._filters = filters or []

    def where(self, filter=None):  # noqa: A002 - mirrors the real signature
        return FakeQuery(self._collection, [*self._filters, filter])

    def stream(self):
        for doc_id, data in self._collection.docs.items():
            if all(self._matches(data, f) for f in self._filters):
                yield FakeSnapshot(doc_id, data, FakeDocument(self._collection, doc_id))

    @staticmethod
    def _matches(data, flt):
        field, op, value = flt.field_path, flt.op_string, flt.value
        actual = data.get(field)
        if op == "==":
            return actual == value
        raise AssertionError(f"unsupported op in fake: {op}")


class FakeCollection(FakeQuery):
    def __init__(self, name):
        self.name = name
        self.docs = {}
        super().__init__(self)

    @property
    def _collection(self):
        return self

    @_collection.setter
    def _collection(self, value):
        pass

    def document(self, doc_id):
        return FakeDocument(self, doc_id)


class FakeBatch:
    def __init__(self):
        self._ops = []

    def delete(self, ref):
        self._ops.append(("delete", ref, None))

    def update(self, ref, data):
        self._ops.append(("update", ref, data))

    def set(self, ref, data, merge=False):
        self._ops.append(("set", ref, data))

    def commit(self):
        for kind, ref, data in self._ops:
            if kind == "delete":
                ref.delete()
            elif kind == "update":
                ref.update(data)
            else:
                ref.set(data)
        self._ops = []


class FakeDb:
    def __init__(self):
        self._collections = {}

    def collection(self, name):
        return self._collections.setdefault(name, FakeCollection(name))

    def batch(self):
        return FakeBatch()


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr("app.repositories.farms.get_db", lambda: db)
    return db


@pytest.fixture
def user():
    return CurrentUser(uid="user-1", email="a@example.com", claims={})


@pytest.fixture
def other_user():
    return CurrentUser(uid="user-2", email="b@example.com", claims={})


def test_create_and_list_farms(fake_db, user):
    from app.repositories import farms as repo

    repo.create_farm(user.uid, "Batı Tarla", "Aydın", None)
    repo.create_farm(user.uid, "Ana Tarla", None, None)

    farms = repo.list_farms(user.uid)
    assert [f["name"] for f in farms] == ["Ana Tarla", "Batı Tarla"]
    assert all(f["owner_uid"] == user.uid for f in farms)


def test_list_farms_is_scoped_to_owner(fake_db, user, other_user):
    from app.repositories import farms as repo

    repo.create_farm(user.uid, "Mine", None, None)
    repo.create_farm(other_user.uid, "Theirs", None, None)

    assert [f["name"] for f in repo.list_farms(user.uid)] == ["Mine"]
    assert [f["name"] for f in repo.list_farms(other_user.uid)] == ["Theirs"]


def test_partial_update_leaves_other_fields_alone(fake_db, user):
    """A PATCH with only `name` must not null out `region`."""
    from app.repositories import farms as repo

    farm = repo.create_farm(user.uid, "Old", "Aydın", None)
    updated = repo.update_farm(
        farm["farm_id"], {"name": "New", "region": None, "location": None}
    )

    assert updated["name"] == "New"
    assert updated["region"] == "Aydın"


def test_delete_farm_cascades_to_plots(fake_db, user):
    from app.repositories import farms as repo

    farm = repo.create_farm(user.uid, "Farm", None, None)
    other = repo.create_farm(user.uid, "Other", None, None)
    repo.create_plot(user.uid, farm["farm_id"], "P1", "Tomato", 1.5, None)
    repo.create_plot(user.uid, farm["farm_id"], "P2", "Potato", None, None)
    keeper = repo.create_plot(user.uid, other["farm_id"], "P3", "Corn", None, None)

    removed = repo.delete_farm(farm["farm_id"])

    assert removed == 2
    assert repo.get_farm(farm["farm_id"]) is None
    assert repo.list_plots(farm["farm_id"]) == []
    # A sibling farm's plots must survive.
    assert [p["plot_id"] for p in repo.list_plots(other["farm_id"])] == [
        keeper["plot_id"]
    ]


def test_plots_are_listed_per_farm(fake_db, user):
    from app.repositories import farms as repo

    a = repo.create_farm(user.uid, "A", None, None)
    b = repo.create_farm(user.uid, "B", None, None)
    repo.create_plot(user.uid, a["farm_id"], "Only A", "Tomato", None, None)
    repo.create_plot(user.uid, b["farm_id"], "Only B", "Corn", None, None)

    assert [p["name"] for p in repo.list_plots(a["farm_id"])] == ["Only A"]
    assert [p["name"] for p in repo.list_plots(b["farm_id"])] == ["Only B"]


def test_owned_or_404_rejects_another_users_record(fake_db, user, other_user):
    """Ownership failure must 404, not 403 — a 403 confirms the id exists."""
    from app.repositories import farms as repo

    farm = repo.create_farm(other_user.uid, "Theirs", None, None)

    with pytest.raises(HTTPException) as exc:
        owned_or_404(repo.get_farm(farm["farm_id"]), user, "farm")
    assert exc.value.status_code == 404


def test_owned_or_404_missing_record(fake_db, user):
    from app.repositories import farms as repo

    with pytest.raises(HTTPException) as exc:
        owned_or_404(repo.get_farm("nope"), user, "farm")
    assert exc.value.status_code == 404
