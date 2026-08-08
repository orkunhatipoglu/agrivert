"""The training/serving contract.

Everything both sides must agree on lives here and NOWHERE else.

This module exists because of a specific failure. `bootstrap_dev_model.py`
was written against `build_label_map`, `DEFAULT_RESIZE` and
`DEFAULT_CENTER_CROP` — three names that never existed. The label-map logic
was copy-pasted inline into the training script, the resize ratio was a magic
`1.14` buried in a transform builder, and the checkpoint format was implied
by whoever wrote `torch.save` last. Nothing declared the contract, so three
different files each invented their own version of it and a new machine hit
`ImportError` on first run.

Rule: if training and serving both need to know something, it is defined
here and imported. Never re-derived, never copy-pasted.
"""

from __future__ import annotations

from typing import Any, TypedDict

# --- Preprocessing ---------------------------------------------------------
# These must match between training and serving or the model sees inputs it
# was never trained on — a silent accuracy loss with no error anywhere.

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_CENTER_CROP: int = 224
# Resize the short side to slightly more than the crop, then centre-crop, so
# the crop isn't padded. 1.14 is the historical ratio (224 -> 255); it is
# named here rather than inlined so serving can't silently disagree.
RESIZE_RATIO: float = 1.14
DEFAULT_RESIZE: int = int(round(DEFAULT_CENTER_CROP * RESIZE_RATIO))

# --- Label vocabulary ------------------------------------------------------
# A raw label is "<Crop>___<Condition>", e.g. "Tomato___Late_blight" or
# "Lettuce___healthy". The separator is three underscores because crop and
# condition names may each contain single underscores.

LABEL_SEPARATOR = "___"
HEALTHY_CONDITION = "healthy"


class LabelEntry(TypedDict):
    """One entry in labels.json. The serving layer reads exactly these keys."""

    raw_label: str
    crop: str
    condition: str
    healthy: bool
    field_coverage: bool


def parse_raw_label(raw_label: str) -> tuple[str, str]:
    """Split a raw label into (crop, condition), both human-readable.

    >>> parse_raw_label("Tomato___Late_blight")
    ('Tomato', 'Late blight')
    >>> parse_raw_label("Pepper,_bell___healthy")
    ('Pepper, bell', 'healthy')
    >>> parse_raw_label("Lettuce")
    ('Lettuce', 'unknown')
    """
    crop, _, condition = raw_label.partition(LABEL_SEPARATOR)
    return (
        crop.replace("_", " ").strip(),
        (condition or "unknown").replace("_", " ").strip(),
    )


def is_healthy(raw_label: str) -> bool:
    """True when the label's condition is the healthy sentinel."""
    _, condition = raw_label.partition(LABEL_SEPARATOR)[0], raw_label.partition(
        LABEL_SEPARATOR
    )[2]
    return condition.strip().lower() == HEALTHY_CONDITION


def build_label_map(
    classes: list[str], field_classes: set[str] | frozenset[str] | None = None
) -> dict[str, LabelEntry]:
    """Build labels.json content: class index -> label metadata.

    `classes` must be in the exact order the model's output layer uses —
    index i in this map is logit i. `field_classes` are the labels with real
    in-the-wild training data; anything outside gets `field_coverage: False`
    so serving can flag it as unvalidated.

    This is THE definition. Training writes it, the dev bootstrap writes it,
    serving reads it. There is no second implementation.
    """
    field = set(field_classes or ())
    label_map: dict[str, LabelEntry] = {}
    for index, raw_label in enumerate(classes):
        crop, condition = parse_raw_label(raw_label)
        label_map[str(index)] = LabelEntry(
            raw_label=raw_label,
            crop=crop,
            condition=condition,
            healthy=is_healthy(raw_label),
            field_coverage=raw_label in field,
        )
    return label_map


# --- Checkpoint format -----------------------------------------------------
# The bug this prevents: bootstrap saved a bare state_dict while predict.py
# loaded ckpt["model"], so a checkpoint that saved fine failed to load.

CHECKPOINT_STATE_KEY = "model"


def wrap_checkpoint(state_dict: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Package a state dict into the checkpoint format serving expects."""
    return {CHECKPOINT_STATE_KEY: state_dict, **extra}


def unwrap_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Get the state dict out of a checkpoint, tolerating a bare one.

    Accepting both shapes is deliberate: older artifacts in the wild were
    saved bare, and failing to load them would strand real trained weights
    over a packaging detail.
    """
    if isinstance(checkpoint, dict) and CHECKPOINT_STATE_KEY in checkpoint:
        return checkpoint[CHECKPOINT_STATE_KEY]
    if isinstance(checkpoint, dict):
        return checkpoint  # already a bare state dict
    raise TypeError(f"unrecognised checkpoint type: {type(checkpoint)!r}")


# --- Artifact set ----------------------------------------------------------
# The three files a model version must ship. The registry, the packager and
# the fetcher all import this rather than each listing them.

REQUIRED_ARTIFACTS: tuple[str, ...] = ("best.pt", "metadata.json", "labels.json")
