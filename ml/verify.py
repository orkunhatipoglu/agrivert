#!/usr/bin/env python3
"""
Pre-flight check for the blended Agrivert pipeline. Run this BEFORE training.

It exercises every part of the path that can fail cheaply — dataset discovery,
class mapping, split integrity, the albumentations pipeline, batch collation,
CUDA, and a single forward+backward step — so that a 30-minute training run
does not die on a path typo or an albumentations API change.

It deliberately does NOT train. One optimizer step on one batch, then stop.

    python verify.py --download
    python verify.py --plantvillage-root ~/data/pv --plantdoc-root ~/data/pd
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PASS, FAIL, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"
_failures: list[str] = []


def section(t: str) -> None:
    print(f"\n=== {t} " + "=" * max(0, 60 - len(t)))


def ok(m: str) -> None:
    print(f"{PASS} {m}")


def warn(m: str) -> None:
    print(f"{WARN} {m}")


def bad(m: str) -> None:
    print(f"{FAIL} {m}")
    _failures.append(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantvillage-root", default=None)
    ap.add_argument("--plantdoc-root", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-images", type=int, default=24)
    args = ap.parse_args()

    # ---------------------------------------------------------------
    section("1. Imports and versions")
    try:
        import numpy as np
        import torch
        import albumentations as A
        import sklearn
        import PIL
        ok(f"numpy {np.__version__} | torch {torch.__version__} | "
           f"albumentations {A.__version__} | sklearn {sklearn.__version__} | "
           f"pillow {PIL.__version__}")
    except Exception as exc:  # noqa: BLE001
        bad(f"Import failed: {exc}")
        traceback.print_exc()
        return 1

    try:
        import data as D
    except Exception as exc:  # noqa: BLE001
        bad(f"Could not import data.py: {exc}")
        traceback.print_exc()
        return 1
    ok("data.py imported")

    # ---------------------------------------------------------------
    section("2. Source files: line endings")
    here = Path(__file__).resolve().parent
    for f in sorted(here.glob("*.py")) + sorted(here.glob("*.txt")):
        raw = f.read_bytes()
        if b"\r\n" in raw:
            bad(f"{f.name} has CRLF line endings — run: sed -i 's/\\r$//' {f.name}")
        else:
            ok(f"{f.name}: LF only ({len(raw)} bytes)")

    # ---------------------------------------------------------------
    section("3. Class mapping")
    try:
        D.validate_mapping()
        ok(f"{len(D.PLANTDOC_TO_PLANTVILLAGE)} PlantDoc->PlantVillage mappings, all targets valid")
        ok(f"{len(D.PLANTVILLAGE_CLASSES)} PlantVillage classes in the canonical label space")
    except Exception as exc:  # noqa: BLE001
        bad(f"Mapping invalid: {exc}")
        return 1

    # ---------------------------------------------------------------
    section("4. Dataset discovery")
    def resolve(flag, slug, label):
        if flag:
            return Path(flag)
        if args.download:
            import kagglehub
            print(f"  downloading {slug} ...")
            return Path(kagglehub.dataset_download(slug))
        bad(f"No {label} root given (use --{label}-root or --download)")
        return None

    pv_root = resolve(args.plantvillage_root, "abdallahalidev/plantvillage-dataset", "plantvillage")
    pd_root = resolve(args.plantdoc_root, "manojkumarcs28/plantdoc-dataset", "plantdoc")
    if pv_root is None or pd_root is None:
        return 1
    ok(f"PlantVillage root: {pv_root}")
    ok(f"PlantDoc     root: {pd_root}")
    for r in (pv_root, pd_root):
        if str(r).startswith("/mnt/") and len(str(r)) > 6 and str(r)[5].isalpha():
            warn(f"{r} is on a Windows drive; training IO will be slow in WSL.")

    # ---------------------------------------------------------------
    section("5. Blend assembly and split integrity")
    try:
        blend = D.build_blend(pv_root, pd_root, seed=42, log=lambda m: print(f"       {m}"))
    except Exception as exc:  # noqa: BLE001
        bad(f"build_blend failed: {exc}")
        traceback.print_exc()
        return 1

    splits, classes = blend["splits"], blend["classes"]
    ok(f"{len(classes)} classes; {len(blend['field_classes'])} have field coverage")
    if blend["plantdoc_report"]["unmapped_folders"]:
        warn(f"Unmapped PlantDoc folders ignored: {blend['plantdoc_report']['unmapped_folders']}")

    named = {k: v for k, v in splits.items() if k != "val_blended"}
    import itertools
    leaked = False
    for a, b in itertools.combinations(named, 2):
        overlap = {str(p) for p, _, _ in named[a]} & {str(p) for p, _, _ in named[b]}
        if overlap:
            bad(f"LEAK between {a} and {b}: {len(overlap)} shared files")
            leaked = True
    if not leaked:
        ok("No file appears in more than one split")

    # Compare against the actual held-out set rather than sniffing for the
    # string "PlantDoc" in the path: find_subdir() matches case-insensitively,
    # so a root extracted as 'plantdoc/' would make a substring check silently
    # pass whether or not a leak existed.
    field_test_paths = {str(p) for p, _, _ in splits["test_field"]}
    leaked_into = [name for name in ("train", "val_field", "val_studio")
                   if field_test_paths & {str(p) for p, _, _ in splits[name]}]
    if leaked_into:
        bad(f"PlantDoc official test images leaked into: {leaked_into}")
    else:
        ok("PlantDoc official test split fully held out as the FIELD test set")

    for name, samples in splits.items():
        doms = {}
        for _, _, d in samples:
            doms[d] = doms.get(d, 0) + 1
        print(f"       {name:12} {len(samples):>6}  {doms}")
    if not splits["test_field"]:
        bad("FIELD test split is empty — the whole point of this run is missing")

    idx_max = max(blend["class_to_idx"].values())
    if idx_max != len(classes) - 1:
        bad(f"class_to_idx max index {idx_max} != {len(classes)-1}")
    else:
        ok(f"class indices are contiguous 0..{idx_max}")

    # ---------------------------------------------------------------
    section("6. Augmentation pipeline construction")
    try:
        train_tf = D.build_train_transform(224, 1.0)
        eval_tf = D.build_eval_transform(224)
    except Exception as exc:  # noqa: BLE001
        bad(f"Transform construction failed: {exc}")
        traceback.print_exc()
        return 1
    print("       train: " + "\n              ".join(D.describe_transform(train_tf)))
    print("       eval : " + " -> ".join(D.describe_transform(eval_tf)))
    required = {"RandomResizedCrop", "ColorJitter", "RandomBrightnessContrast"}
    built = " ".join(D.describe_transform(train_tf))
    missing = [r for r in required if r not in built]
    if missing:
        bad(f"Requested transforms missing from the pipeline: {missing}")
    else:
        ok("RandomResizedCrop / ColorJitter / RandomBrightnessContrast present")
    for label, alts in (("ShiftScaleRotate", ("ShiftScaleRotate", "Affine")),
                        ("Cutout/CoarseDropout", ("CoarseDropout", "Cutout")),
                        ("MotionBlur", ("MotionBlur",))):
        if any(a in built for a in alts):
            ok(f"{label} present (as {[a for a in alts if a in built][0]})")
        else:
            bad(f"{label} missing from the pipeline")

    # ---------------------------------------------------------------
    section("7. Real image decode + transform")
    import numpy as np
    import torch

    field = [s for s in splits["train"] if s[2] == D.DOMAIN_FIELD][: args.sample_images // 2]
    studio = [s for s in splits["train"] if s[2] == D.DOMAIN_STUDIO][: args.sample_images // 2]
    checked = 0
    for path, cls, dom in field + studio:
        try:
            arr = D.load_image_rgb(path)
            assert arr.ndim == 3 and arr.shape[2] == 3, f"bad shape {arr.shape}"
            assert arr.dtype == np.uint8, f"bad dtype {arr.dtype}"
            t = train_tf(image=arr)["image"]
            e = eval_tf(image=arr)["image"]
            assert tuple(t.shape) == (3, 224, 224), f"train tensor {tuple(t.shape)}"
            assert tuple(e.shape) == (3, 224, 224), f"eval tensor {tuple(e.shape)}"
            assert t.dtype == torch.float32, f"train dtype {t.dtype}"
            checked += 1
        except Exception as exc:  # noqa: BLE001
            bad(f"{dom} image failed: {path} -> {exc}")
            break
    if checked:
        ok(f"{checked} images decoded and transformed; tensors are float32 [3,224,224]")
        sample = train_tf(image=D.load_image_rgb(field[0][0] if field else studio[0][0]))["image"]
        ok(f"normalized range: min {sample.min():.2f} max {sample.max():.2f} "
           f"mean {sample.mean():.2f} (expect roughly -2.5..2.7, mean near 0)")

    # ---------------------------------------------------------------
    section("8. Dataset + DataLoader batch generation")
    from torch.utils.data import DataLoader, WeightedRandomSampler
    ds = D.BlendedDataset(splits["train"], blend["class_to_idx"], train_tf)
    ok(f"BlendedDataset len {len(ds)}")
    img, tgt, dom = ds[0]
    ok(f"single item: image {tuple(img.shape)} target {tgt} domain {dom}")
    if not (0 <= tgt < len(classes)):
        bad(f"target {tgt} out of range for {len(classes)} classes")

    w = D.build_sample_weights([c for _, c, _ in splits["train"]],
                               [d for _, _, d in splits["train"]], 8.0)
    share = D.effective_domain_share(w, [d for _, _, d in splits["train"]])
    ok(f"field share per epoch after 8x oversampling: {share:.1%}")
    if share < 0.15:
        warn("Field share is low; consider raising --field-oversample")

    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(w), replacement=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    try:
        images, targets, domains = next(iter(loader))
    except Exception as exc:  # noqa: BLE001
        bad(f"DataLoader batch failed (workers={args.workers}): {exc}")
        traceback.print_exc()
        return 1
    ok(f"batch: images {tuple(images.shape)} {images.dtype} | "
       f"targets {tuple(targets.shape)} | field in batch {int((domains==1).sum())}/{len(domains)}")
    if int(targets.min()) < 0 or int(targets.max()) >= len(classes):
        bad(f"target index out of range: {int(targets.min())}..{int(targets.max())}")
    else:
        ok(f"target indices in range 0..{len(classes)-1}")

    # ---------------------------------------------------------------
    section("9. CUDA")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        ok(f"{p.name} | {p.total_memory/1e9:.1f} GB | torch cuda {torch.version.cuda} "
           f"| bf16 {torch.cuda.is_bf16_supported()}")
        device = torch.device("cuda")
    else:
        if torch.version.cuda is None:
            bad("CPU-only torch build installed. Reinstall:\n"
                "      pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu124")
        else:
            bad("CUDA-enabled torch but no visible GPU — update the Windows NVIDIA "
                "driver and run `wsl --shutdown` from PowerShell.")
        warn("Continuing the forward-pass check on CPU.")
        device = torch.device("cpu")

    # ---------------------------------------------------------------
    section("10. One forward + backward step (no training)")
    try:
        from train_mobilenetv2 import build_model, freeze_bn_stats, set_backbone_trainable
        import torch.nn as nn

        model = build_model(len(classes)).to(device)
        set_backbone_trainable(model, 7)
        model.train()
        freeze_bn_stats(model, 7)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            images = images.to(memory_format=torch.channels_last)

        images, targets = images.to(device), targets.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = crit(logits, targets)
        ok(f"forward: logits {tuple(logits.shape)} loss {loss.item():.4f}")
        if logits.shape[1] != len(classes):
            bad(f"model outputs {logits.shape[1]} logits for {len(classes)} classes")

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total = sum(1 for p in model.parameters() if p.requires_grad)
        ok(f"backward + optimizer step completed ({grads}/{total} trainable tensors got grads)")
        if grads == 0:
            bad("No gradients produced — nothing would train")

        frozen_bn = [m for i, b in enumerate(model.features) if i < 7
                     for m in b.modules() if isinstance(m, nn.BatchNorm2d)]
        if frozen_bn and all(not m.training for m in frozen_bn):
            ok(f"{len(frozen_bn)} BatchNorm layers in frozen blocks are held in eval mode")
        elif frozen_bn:
            bad("BatchNorm in frozen blocks is still in train mode")

        if device.type == "cuda":
            ok(f"peak VRAM this step: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    except Exception as exc:  # noqa: BLE001
        bad(f"Forward/backward failed: {exc}")
        traceback.print_exc()

    # ---------------------------------------------------------------
    section("Result")
    if _failures:
        print(f"{FAIL} {len(_failures)} problem(s):")
        for f in _failures:
            print(f"        - {f}")
        return 1
    print(f"{PASS} All checks passed. Ready to train:")
    print("\n      python train_mobilenetv2.py --download --out-dir ./artifacts\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
