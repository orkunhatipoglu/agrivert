"""Model construction — one definition, used by training, serving and the
dev bootstrap.

`bootstrap_dev_model.py` tried to import `_build_backbone` from `predict.py`,
where it did not exist; `predict.py` built its classifier head inline and the
training script built the same head a second time. Three copies of "what is
the architecture" is exactly how a placeholder model ends up structurally
incompatible with the loader.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml.contract import unwrap_checkpoint, wrap_checkpoint

# Head dropout. Declared here because the head shape is part of the
# checkpoint's structure: change it and every existing best.pt fails to load.
HEAD_DROPOUT = 0.3

SUPPORTED_ARCHITECTURES = ("mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0")


def build_backbone(
    architecture: str, num_classes: int, pretrained: bool = False
) -> nn.Module:
    """Build the model, classifier head replaced for `num_classes`.

    `pretrained=False` is the default because loading a checkpoint overwrites
    the weights anyway — downloading ImageNet weights just to discard them
    wastes a few hundred MB on every serving cold start.
    """
    from torchvision import models

    if architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT), nn.Linear(in_features, num_classes)
        )
        return model

    if architecture == "mobilenet_v3_large":
        weights = (
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT), nn.Linear(in_features, num_classes)
        )
        return model

    raise ValueError(
        f"unsupported architecture {architecture!r}; "
        f"expected one of {', '.join(SUPPORTED_ARCHITECTURES)}"
    )


def save_checkpoint(path, model: nn.Module, **extra: Any) -> None:
    """Save in the format the serving loader expects.

    Unwraps torch.compile and DataParallel first, so a checkpoint never
    carries `_orig_mod.` or `module.` key prefixes that the plain model then
    refuses to load.
    """
    inner = getattr(model, "_orig_mod", model)
    inner = getattr(inner, "module", inner)
    torch.save(wrap_checkpoint(inner.state_dict(), **extra), path)


def load_checkpoint_into(model: nn.Module, path, map_location="cpu") -> nn.Module:
    """Load weights saved by `save_checkpoint` (or a bare state dict)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(unwrap_checkpoint(checkpoint))
    return model


def _head_keys(state: dict) -> tuple[str, str] | None:
    """The (weight, bias) keys of the final Linear, whatever the backbone."""
    for w in ("classifier.1.weight", "classifier.3.weight"):
        if w in state:
            return w, w.replace("weight", "bias")
    return None


def warm_start_head(
    model: nn.Module,
    path,
    old_classes: list[str],
    new_classes: list[str],
    map_location="cpu",
) -> tuple[int, list[str]]:
    """Load a checkpoint trained on `old_classes` into a model for `new_classes`.

    The backbone transfers wholesale. The head cannot: it has one row per
    class, so a taxonomy that gained a class has a differently shaped final
    Linear and `load_state_dict` refuses it outright.

    Rows are matched **by class name**, never by position. Matching by index
    would appear to work and be quietly wrong the moment a class is inserted
    anywhere but the end — every subsequent class would inherit another
    class's learned weights, and the only symptom would be a model that
    trains from a worse-than-random start for reasons nothing logs.

    Classes with no counterpart in the checkpoint keep their fresh
    initialisation. Returns (rows transferred, names of the new classes).
    """
    checkpoint = unwrap_checkpoint(
        torch.load(path, map_location=map_location, weights_only=False)
    )
    keys = _head_keys(checkpoint)
    target = model.state_dict()
    if keys is None or _head_keys(target) != keys:
        raise ValueError(
            f"cannot locate a matching classifier head in {path}; "
            "warm-starting across different architectures is not supported"
        )
    w_key, b_key = keys
    old_w, old_b = checkpoint[w_key], checkpoint[b_key]
    if old_w.shape[0] != len(old_classes):
        raise ValueError(
            f"{path} has a {old_w.shape[0]}-row head but was described as "
            f"{len(old_classes)} classes — refusing to guess the alignment"
        )

    new_w, new_b = target[w_key].clone(), target[b_key].clone()
    old_index = {name: i for i, name in enumerate(old_classes)}
    transferred = 0
    for i, name in enumerate(new_classes):
        j = old_index.get(name)
        if j is not None:
            new_w[i], new_b[i] = old_w[j], old_b[j]
            transferred += 1

    checkpoint[w_key], checkpoint[b_key] = new_w, new_b
    model.load_state_dict(checkpoint)
    return transferred, [c for c in new_classes if c not in old_index]
