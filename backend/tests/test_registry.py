"""Model registry tests — the swap mechanism."""

from __future__ import annotations

import pytest

from app.ml import registry
from app.ml.registry import ModelRegistryError


def test_discovers_complete_version(registry_settings):
    versions = registry.discover_versions()
    assert [v.version for v in versions] == ["v1-test"]


def test_skips_incomplete_version(registry_settings, model_dir):
    """A dir missing best.pt must not be offered as usable."""
    broken = model_dir.parent / "v2-broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{}")
    (broken / "labels.json").write_text("{}")

    assert [v.version for v in registry.discover_versions()] == ["v1-test"]

    with pytest.raises(ModelRegistryError, match="incomplete"):
        registry.get_version("v2-broken")


def test_rejects_path_traversal(registry_settings):
    for bad in ("../etc", "a/b", ".hidden"):
        with pytest.raises(ModelRegistryError, match="invalid version name"):
            registry.get_version(bad)


def test_missing_version_raises(registry_settings):
    with pytest.raises(ModelRegistryError, match="not found on disk"):
        registry.get_version("nope")


def test_summary_reads_metrics(registry_settings):
    summary = registry.get_version("v1-test").summary()
    assert summary["version"] == "v1-test"
    assert summary["num_classes"] == 38
    assert summary["confidence_threshold"] == 0.95
    assert summary["metrics"]["test_field"]["accuracy"] == pytest.approx(0.6525)


def test_falls_back_to_sole_version(registry_settings, monkeypatch):
    """With Firestore unavailable and one version on disk, use it."""
    monkeypatch.setattr(
        "app.repositories.models.get_active_version_name",
        lambda: (_ for _ in ()).throw(RuntimeError("firestore down")),
    )
    assert registry.resolve_active_version().version == "v1-test"


def test_ambiguous_without_active_raises(registry_settings, model_dir, monkeypatch):
    """Two versions and no active pointer must fail loudly, not guess.

    Silently picking one would mean serving an unknown model.
    """
    second = model_dir.parent / "v2-test"
    second.mkdir()
    for name in ("best.pt", "metadata.json", "labels.json"):
        (second / name).write_bytes((model_dir / name).read_bytes())

    monkeypatch.setattr(
        "app.repositories.models.get_active_version_name", lambda: None
    )
    with pytest.raises(ModelRegistryError, match="none is active"):
        registry.resolve_active_version()
