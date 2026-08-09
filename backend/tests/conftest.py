"""Test fixtures.

Firebase and torch are both mocked out. These tests cover the logic this
scaffold actually owns — validation rules, registry resolution, schema
contracts — not Google's SDK or PyTorch.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def jpeg_bytes():
    """A small valid JPEG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (34, 139, 34)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_jpeg_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (34, 139, 34)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def model_dir(tmp_path):
    """A complete, fake model version directory."""
    d = tmp_path / "models" / "v1-test"
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"not-a-real-checkpoint")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "mobilenetv2-agrivert-blended",
                "architecture": "mobilenet_v2",
                "num_classes": 38,
                "best_epoch": 8,
                "classes": ["Apple___Apple_scab", "Tomato___healthy"],
                "field_covered_classes": ["Apple___Apple_scab"],
                "calibration": {
                    "temperature": 0.85,
                    "recommended_confidence_threshold": 0.95,
                },
                "metrics": {
                    "test_field": {"accuracy": 0.6525, "macro_f1": 0.6438},
                    "test_studio": {"accuracy": 0.9871, "macro_f1": 0.9836},
                },
                "preprocessing": {"center_crop": 224},
            }
        )
    )
    (d / "labels.json").write_text(
        json.dumps(
            {
                "0": {
                    "raw_label": "Apple___Apple_scab",
                    "crop": "Apple",
                    "condition": "Apple scab",
                    "healthy": False,
                    "field_coverage": True,
                },
                "1": {
                    "raw_label": "Tomato___healthy",
                    "crop": "Tomato",
                    "condition": "healthy",
                    "healthy": True,
                    "field_coverage": True,
                },
            }
        )
    )
    return d


@pytest.fixture
def registry_settings(model_dir, monkeypatch):
    """Point the registry at the fake model dir."""
    from app.config import Settings, get_settings

    get_settings.cache_clear()
    # _env_file=None keeps the developer's backend/.env out of the fixture.
    # Without it, a local DEFAULT_MODEL_VERSION leaks in and the fallback
    # tests below resolve to a version that doesn't exist in the tmp dir.
    settings = Settings(model_registry_dir=model_dir.parent, _env_file=None)
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.ml.registry.get_settings", lambda: settings)
    yield settings
    get_settings.cache_clear()
