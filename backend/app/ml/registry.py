"""Versioned model registry.

Layout on disk:

    backend/models/
        v1-blended-20260808/
            best.pt
            metadata.json
            labels.json
        v2-.../
            ...

A version is *usable* if its directory holds all three required files. Which
version serves traffic is recorded in Firestore (`model_versions`, the doc
with `active: true`), so `POST /admin/models/{version}/activate` is a data
change, not a redeploy. `DEFAULT_MODEL_VERSION` (or, failing that, the single
available version) is the local-dev fallback when Firestore has no record.

Swapping in a retrained model is therefore: drop a new dir in models/,
register it, activate it. No code changes — which is the requirement that
drove this design.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)

REQUIRED_FILES = ("best.pt", "metadata.json", "labels.json")


class ModelRegistryError(RuntimeError):
    """Registry could not resolve a usable model version."""


@dataclass(frozen=True)
class ModelVersion:
    """A resolved, on-disk model version."""

    version: str
    path: Path

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    def load_metadata(self) -> dict:
        return json.loads(self.metadata_path.read_text())

    def summary(self) -> dict:
        """Version + eval metrics, for GET /admin/models.

        Reads metadata.json rather than caching, so a hand-edited threshold is
        reflected without a restart.
        """
        meta = self.load_metadata()
        return {
            "version": self.version,
            "model_name": meta.get("model_name"),
            "architecture": meta.get("architecture"),
            "num_classes": meta.get("num_classes"),
            "best_epoch": meta.get("best_epoch"),
            # test_field is the number that matters; test_studio flatters the
            # model badly (project_context.md §3 step 5).
            "metrics": meta.get("metrics", {}),
            "confidence_threshold": meta.get("calibration", {}).get(
                "recommended_confidence_threshold"
            ),
            "temperature": meta.get("calibration", {}).get("temperature"),
            "caveat": meta.get("caveat"),
        }


def _registry_dir() -> Path:
    return Path(get_settings().model_registry_dir)


def discover_versions() -> list[ModelVersion]:
    """Every complete model version on disk, newest-name-last."""
    root = _registry_dir()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        missing = [f for f in REQUIRED_FILES if not (child / f).is_file()]
        if missing:
            log.warning(
                "skipping model dir %s: missing %s", child.name, ", ".join(missing)
            )
            continue
        found.append(ModelVersion(version=child.name, path=child))
    return found


def get_version(version: str) -> ModelVersion:
    """Resolve one version by name, rejecting path traversal."""
    if not version or "/" in version or "\\" in version or version.startswith("."):
        raise ModelRegistryError(f"invalid version name: {version!r}")
    path = _registry_dir() / version
    if not path.is_dir():
        raise ModelRegistryError(f"model version not found on disk: {version}")
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise ModelRegistryError(
            f"model version {version} is incomplete; missing: {', '.join(missing)}"
        )
    return ModelVersion(version=version, path=path)


def resolve_active_version() -> ModelVersion:
    """The version that should serve traffic.

    Order: Firestore active record -> DEFAULT_MODEL_VERSION -> the only
    version on disk. Raises if none of those yields a usable version, because
    serving a silently-wrong model is worse than failing to start.
    """
    from app.repositories.models import get_active_version_name  # local: avoid cycle

    name = None
    try:
        name = get_active_version_name()
    except Exception as exc:  # Firestore unreachable / not configured
        log.warning("could not read active model version from Firestore: %s", exc)

    if not name:
        name = get_settings().default_model_version

    if not name:
        available = discover_versions()
        if len(available) == 1:
            log.info("defaulting to the only model version present: %s", available[0].version)
            return available[0]
        if not available:
            raise ModelRegistryError(
                f"no usable model versions in {_registry_dir()}. "
                "Register one with scripts/register_model.py."
            )
        raise ModelRegistryError(
            "multiple model versions present and none is active; set "
            f"DEFAULT_MODEL_VERSION or activate one. Found: "
            f"{', '.join(v.version for v in available)}"
        )

    return get_version(name)
