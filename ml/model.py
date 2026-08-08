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
