"""Guards on the training/serving contract.

These exist because of a real incident: a fresh checkout failed with
`ImportError: cannot import name 'DEFAULT_CENTER_CROP' from 'data'`.
`bootstrap_dev_model.py` imported three names that had never existed, plus a
`_build_backbone` that lived nowhere, and saved a bare `state_dict()` that
the serving loader could not read.

Each test below maps to one of those breakages. They are cheap and they fail
loudly the moment someone re-splits the contract across two files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_contract_exports_the_names_bootstrap_needs():
    """The three names whose absence broke a teammate's first run."""
    from ml.contract import (
        DEFAULT_CENTER_CROP,
        DEFAULT_RESIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_label_map,
    )

    assert DEFAULT_CENTER_CROP == 224
    assert DEFAULT_RESIZE == 255  # round(224 * 1.14)
    assert len(IMAGENET_MEAN) == 3 and len(IMAGENET_STD) == 3
    assert callable(build_label_map)


def test_build_backbone_is_importable_and_shared():
    from ml.model import build_backbone

    model = build_backbone("mobilenet_v2", 30)
    assert model.classifier[-1].out_features == 30


def test_label_map_matches_serving_expectations():
    """predict.py reads exactly these keys off each entry."""
    from ml.contract import build_label_map

    label_map = build_label_map(
        ["Tomato___Late_blight", "Lettuce___healthy"], {"Tomato___Late_blight"}
    )
    entry = label_map["0"]
    assert set(entry) == {
        "raw_label", "crop", "condition", "healthy", "field_coverage",
    }
    assert entry["crop"] == "Tomato"
    assert entry["condition"] == "Late blight"
    assert entry["healthy"] is False
    assert entry["field_coverage"] is True

    healthy = label_map["1"]
    assert healthy["healthy"] is True
    assert healthy["field_coverage"] is False


def test_checkpoint_roundtrip_survives_the_loader():
    """A bare state_dict saved by one side must load on the other.

    This is the mismatch that would have bitten immediately after the
    ImportError was fixed.
    """
    import torch

    from ml.contract import unwrap_checkpoint, wrap_checkpoint
    from ml.model import build_backbone, load_checkpoint_into, save_checkpoint

    model = build_backbone("mobilenet_v2", 5)
    tmp = Path("/tmp/_contract_ckpt.pt")
    try:
        save_checkpoint(tmp, model, epoch=3)
        reloaded = build_backbone("mobilenet_v2", 5)
        load_checkpoint_into(reloaded, tmp)
    finally:
        tmp.unlink(missing_ok=True)

    # A bare state dict (the old bootstrap's output) must still be accepted.
    assert unwrap_checkpoint(model.state_dict()) is not None
    assert "model" in wrap_checkpoint({"w": 1})


def test_eval_transform_uses_the_contract_constants():
    """Serving preprocessing must be derived from the contract, not a magic
    number copied into a transform builder."""
    pytest.importorskip("albumentations")
    from ml.contract import DEFAULT_CENTER_CROP, DEFAULT_RESIZE
    from ml.data import build_eval_transform

    names = [t.__class__.__name__ for t in build_eval_transform(DEFAULT_CENTER_CROP)]
    assert "SmallestMaxSize" in names and "CenterCrop" in names
    assert DEFAULT_RESIZE > DEFAULT_CENTER_CROP  # crop is never padded


def test_taxonomy_is_self_consistent():
    from ml.taxonomy import validate_taxonomy

    validate_taxonomy()  # raises on a typo or an unreachable class


def test_uncovered_crops_are_declared_not_silently_missing():
    """The product asks for crops no dataset provides. That must stay visible
    in code, not live in someone's memory."""
    from ml.taxonomy import UNCOVERED_TARGETS, VERTICAL_CLASSES

    crops = {c.partition("___")[0] for c in VERTICAL_CLASSES}
    for missing in UNCOVERED_TARGETS:
        assert missing not in crops, (
            f"{missing} is listed as uncovered but a class exists for it — "
            "update UNCOVERED_TARGETS"
        )
