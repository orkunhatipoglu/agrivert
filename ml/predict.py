#!/usr/bin/env python3
"""
Inference helper for the Agrivert diagnosis service (blended-model version).

Preprocessing is imported from data.py rather than reimplemented, so serving
cannot drift away from training. Everything variable — normalization, crop
size, calibration temperature, confidence threshold — is read from
metadata.json, so retraining updates serving automatically.

CLI:
    python predict.py --artifacts ./artifacts --image leaf.jpg

Service:
    from predict import DiseaseClassifier
    clf = DiseaseClassifier("./artifacts")   # once at startup
    result = clf.predict(image_bytes)        # per request
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml import data as D
from ml.contract import build_label_map  # noqa: F401  (re-exported for callers)
from ml.model import build_backbone, load_checkpoint_into


class DiseaseClassifier:
    def __init__(
        self,
        artifacts_dir: str | Path,
        device: str | None = None,
        tta: bool = False,
    ):
        self.dir = Path(artifacts_dir)
        self.metadata = json.loads((self.dir / "metadata.json").read_text())
        self.labels = json.loads((self.dir / "labels.json").read_text())
        self.classes: list[str] = self.metadata["classes"]
        self.field_classes = set(self.metadata.get("field_covered_classes", []))

        cal = self.metadata["calibration"]
        self.temperature = float(cal["temperature"])
        self.threshold = float(cal["recommended_confidence_threshold"])
        # What that gate is actually worth, measured on data the threshold was
        # NOT fitted on. Null for models calibrated before this was recorded —
        # callers must treat "unknown" as unknown rather than as "fine".
        verified = cal.get("verified") or {}
        self.expected_accuracy = verified.get("selective_accuracy")
        self.expected_accuracy_95ci = verified.get("selective_accuracy_95ci")
        self.meets_target = cal.get("meets_target")
        self.target_accuracy = cal.get("target_selective_accuracy")

        pre = self.metadata["preprocessing"]
        self.tta = tta
        self.transform = D.build_eval_transform(pre["center_crop"])
        self.tta_transforms = (
            D.build_tta_transforms(pre["center_crop"]) if tta else None
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        architecture = self.metadata.get("architecture", "mobilenet_v2")
        model = build_backbone(architecture, len(self.classes))
        load_checkpoint_into(model, self.dir / "best.pt")
        self.model = model.eval().to(self.device)

    def _to_array(self, image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (bytes, bytearray)):
            from PIL import Image, ImageOps
            with Image.open(io.BytesIO(image)) as im:
                return np.array(ImageOps.exif_transpose(im).convert("RGB"))
        return D.load_image_rgb(image)

    @torch.no_grad()
    def predict(self, image, top_k: int = 3) -> dict:
        arr = self._to_array(image)
        if self.tta_transforms:
            # Average the calibrated probabilities, not the logits: temperature
            # was fitted on a single view's softmax, so it must be applied per
            # view for the averaged score to stay on the threshold's scale.
            views = [
                F.softmax(
                    self.model(
                        tf(image=(arr[:, ::-1].copy() if flip else arr))["image"]
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    / self.temperature,
                    dim=1,
                )[0]
                for tf, flip in self.tta_transforms
            ]
            probs = torch.stack(views).mean(0).float().cpu()
        else:
            tensor = self.transform(image=arr)["image"].unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits / self.temperature, dim=1)[0].float().cpu()

        conf, idx = probs.max(0)
        conf, idx = float(conf), int(idx)
        meta = self.labels[str(idx)]
        certain = conf >= self.threshold

        top_probs, top_idx = probs.topk(min(top_k, len(self.classes)))
        return {
            # Feeds the API's verdict states directly. Below threshold must NOT
            # surface as a diagnosis — a confident wrong answer to a farmer is
            # worse than an honest "I can't tell from this photo".
            "status": "completed" if certain else "uncertain",
            "crop": meta["crop"] if certain else None,
            "condition": meta["condition"] if certain else None,
            "healthy": meta["healthy"] if certain else None,
            "raw_label": meta["raw_label"] if certain else None,
            "confidence": round(conf, 4),
            "threshold": self.threshold,
            # How often an ACCEPTED prediction is right, measured on held-out
            # data the threshold was not fitted on. This is the number a caller
            # should show a grower — `confidence` is this photo's score, not a
            # success rate, and reading it as one overstates the model badly.
            "expected_accuracy": self.expected_accuracy,
            "expected_accuracy_95ci": self.expected_accuracy_95ci,
            # False => the model misses its own declared quality bar. Callers
            # must hedge harder and must not advertise the target accuracy.
            "meets_target": self.meets_target,
            # False => the model saw zero real field photos of this class in
            # training, so treat the prediction with extra suspicion.
            "field_validated": bool(meta["raw_label"] in self.field_classes),
            "model_version": self.metadata.get("model_name", "unknown"),
            "alternatives": [
                {
                    "raw_label": self.labels[str(int(i))]["raw_label"],
                    "crop": self.labels[str(int(i))]["crop"],
                    "condition": self.labels[str(int(i))]["condition"],
                    "confidence": round(float(p), 4),
                }
                for p, i in zip(top_probs, top_idx)
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--image", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Average 3 zoom levels. Steadier confidence when framing varies "
            "(class flips under a small re-crop drop from ~31%% to ~20%%), at "
            "3x inference cost and NO accuracy gain."
        ),
    )
    args = ap.parse_args()
    clf = DiseaseClassifier(args.artifacts, tta=args.tta)
    print(json.dumps(clf.predict(args.image, args.top_k), indent=2))


if __name__ == "__main__":
    main()
