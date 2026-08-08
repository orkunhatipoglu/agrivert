#!/usr/bin/env python3
"""
Fine-tune MobileNetV2 on a BLENDED PlantVillage + PlantDoc dataset for Agrivert.

What changed versus the PlantVillage-only version, and why
---------------------------------------------------------
The old pipeline hit ~99% on the PlantVillage test split and would still have
been unreliable in a field. PlantVillage is studio photography — detached leaf,
uniform background, even light — so a large share of what the network learned
was the *background*, not the lesion. Three changes address that:

  1. PlantDoc (real in-field phone photos) is blended in and heavily
     oversampled, so field images are ~37% of what the model sees per epoch
     instead of the ~4% their raw count would give.
  2. Aggressive albumentations augmentation, with CoarseDropout specifically to
     punch holes in the image so no single region (least of all a clean
     background) can carry the prediction.
  3. Evaluation is split by domain. There is a STUDIO test number and a FIELD
     test number and they are never averaged together. Checkpoint selection,
     calibration and the confidence threshold all key off the FIELD split,
     because that is the distribution the app actually sees.

Expect the field number to be far below the studio number. That gap is the
honest measure of this project, and shrinking it is the work.

Outputs (--out-dir):
    best.pt, last.pt                     checkpoints
    mobilenetv2_agrivert.onnx            deployable model
    labels.json                          idx -> {crop, condition, healthy}
    metadata.json                        preprocessing, metrics, calibration
    confusion_matrix_field.csv           held-out PlantDoc test
    confusion_matrix_studio.csv          held-out PlantVillage test
    classification_report_field.txt      per-class precision / recall / F1
    classification_report_studio.txt
    per_class_metrics.csv                precision/recall/F1/support, both domains
    history.json

Run:
    python train_mobilenetv2.py --download --out-dir ./artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import data as D

PLANTVILLAGE_SLUG = "abdallahalidev/plantvillage-dataset"
PLANTDOC_SLUG = "manojkumarcs28/plantdoc-dataset"


@dataclass
class Config:
    out_dir: str
    image_size: int = D.IMAGE_SIZE
    batch_size: int = 64
    workers: int = 8
    seed: int = 42

    head_epochs: int = 3
    finetune_epochs: int = 17
    unfreeze_from: int = 7          # unfreeze deeper than before: adapting to a
                                    # new visual domain needs more than the head
    head_lr: float = 1e-3
    backbone_lr: float = 1.5e-4
    head_lr_phase2: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    grad_clip: float = 1.0

    field_oversample: float = 8.0
    aug_strength: float = 1.0
    balance_classes: bool = True
    early_stop_patience: int = 6
    select_on: str = "field"

    amp_dtype: str = "fp16"
    channels_last: bool = True
    compile_model: bool = False

    target_selective_accuracy: float = 0.90


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn_about_wsl_paths(path: Path) -> None:
    p = str(path)
    if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
        log("=" * 78)
        log("WARNING: data lives on a Windows drive mounted into WSL (/mnt/...).")
        log("Reading many small images over drvfs will bottleneck the GPU badly.")
        log("Move it onto the Linux filesystem (e.g. ~/datasets) for a big speedup.")
        log("=" * 78)


def download(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub not installed. `pip install kagglehub` or pass explicit roots.")
    log(f"Downloading {slug} via kagglehub (cached after first run)...")
    path = Path(kagglehub.dataset_download(slug))
    log(f"  -> {path}")
    return path


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, unfreeze_from: int | None) -> None:
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_from is not None:
        for block in model.features[unfreeze_from:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def freeze_bn_stats(model: nn.Module, unfreeze_from: int | None) -> None:
    """Hold BatchNorm in eval mode wherever weights are frozen.

    Extra important with a blended dataset: PlantDoc's colour/lighting
    statistics differ sharply from PlantVillage's, so letting BN running stats
    drift in frozen blocks corrupts the pretrained features from epoch one.
    """
    for i, block in enumerate(model.features):
        if unfreeze_from is None or i < unfreeze_from:
            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro F1 over classes actually present in `labels`.

    Averaging over all 38 classes would permanently drag the field score down,
    since the field splits only cover 28 of them — a constant penalty that
    tells us nothing and makes runs incomparable.
    """
    preds = logits.argmax(1)
    present = torch.unique(labels)
    f1s = []
    for c in present.tolist():
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_table(logits, labels, classes):
    from sklearn.metrics import precision_recall_fscore_support
    preds = logits.argmax(1).numpy()
    y = labels.numpy()
    idx = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(
        y, preds, labels=idx, zero_division=0)
    return [
        {"class": classes[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f[i]), "support": int(s[i])}
        for i in idx
    ]


def write_reports(out_dir: Path, tag: str, logits, labels, classes):
    from sklearn.metrics import classification_report, confusion_matrix
    preds = logits.argmax(1).numpy()
    y = labels.numpy()
    present = sorted(set(y.tolist()) | set(preds.tolist()))
    (out_dir / f"classification_report_{tag}.txt").write_text(
        classification_report(y, preds, labels=present,
                              target_names=[classes[i] for i in present],
                              digits=4, zero_division=0))
    cm = confusion_matrix(y, preds, labels=present)
    with open(out_dir / f"confusion_matrix_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = [classes[i] for i in present]
        w.writerow([f"true\\pred ({tag})"] + names)
        for name, row in zip(names, cm):
            w.writerow([name] + row.tolist())


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def threshold_table(logits, labels, temperature,
                    targets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    probs = F.softmax(logits / temperature, dim=1)
    conf, preds = probs.max(1)
    correct = preds == labels
    rows = []
    for t in targets:
        keep = conf >= t
        n = int(keep.sum())
        rows.append({
            "threshold": float(t),
            "coverage": float(keep.float().mean()),
            "n_accepted": n,
            # None, not NaN: these rows go into metadata.json, and bare NaN is
            # not valid JSON (RFC 8259). Python's json.dumps emits it happily
            # but any strict parser on the serving side would reject the file.
            "selective_accuracy": float(correct[keep].float().mean()) if n else None,
        })
    return rows


def pick_threshold(rows, target_acc: float, min_coverage: float = 0.15) -> float:
    ok = [r for r in rows
          if r["selective_accuracy"] is not None
          and r["selective_accuracy"] >= target_acc
          and r["coverage"] >= min_coverage]
    if ok:
        return float(min(r["threshold"] for r in ok))
    # Nothing reaches the target: take the strictest option that still keeps a
    # usable share of predictions, rather than silently shipping a threshold
    # that rejects everything.
    usable = [r for r in rows if r["coverage"] >= min_coverage]
    return float(max(r["threshold"] for r in usable)) if usable else float(rows[0]["threshold"])


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def autocast_ctx(cfg: Config, device: torch.device):
    if cfg.amp_dtype == "off" or device.type != "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, cfg, device,
              unfreeze_from, desc, max_steps: int = 0):
    model.train(True)
    freeze_bn_stats(model, unfreeze_from)
    total_loss = total_correct = total_n = 0
    field_n = 0
    t0 = time.time()
    for step, (images, labels, domains) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with autocast_ctx(cfg, device):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(1) == labels).sum())
        total_n += bs
        field_n += int((domains == 1).sum())

        if step % 50 == 0:
            log(f"  {desc} step {step}/{len(loader)} loss {total_loss/total_n:.4f} "
                f"acc {total_correct/total_n:.4f} field {field_n/total_n:.2f} "
                f"({total_n/max(time.time()-t0,1e-6):.0f} img/s)")
        if max_steps and (step + 1) >= max_steps:
            log(f"  {desc} stopped early at {max_steps} steps (--max-steps-per-epoch)")
            break
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1), field_n / max(total_n, 1)


@torch.no_grad()
def collect_logits(model, loader, cfg, device):
    model.eval()
    L, Y = [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.to(memory_format=torch.channels_last)
        with autocast_ctx(cfg, device):
            logits = model(images)
        L.append(logits.float().cpu())
        Y.append(labels)
    return torch.cat(L), torch.cat(Y)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on PlantVillage + PlantDoc")
    ap.add_argument("--plantvillage-root", default=None)
    ap.add_argument("--plantdoc-root", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--image-size", type=int, default=D.IMAGE_SIZE)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=17)
    ap.add_argument("--unfreeze-from", type=int, default=7)
    ap.add_argument("--field-oversample", type=float, default=8.0,
                    help="Sampling boost for PlantDoc images (1.0 = none)")
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--select-on", choices=["field", "blended", "studio"], default="field",
                    help="Which val split drives checkpoint selection")
    ap.add_argument("--restrict-to-field-classes", action="store_true",
                    help="Train only the 28 classes PlantDoc covers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16", "off"], default="fp16")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--verify-images", action="store_true",
                    help="Pre-scan and drop undecodable files (slow first pass, cached)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0,
                    help="Debug only: cap training steps per epoch")
    args = ap.parse_args()

    if args.workers is None:
        args.workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    cfg = Config(
        out_dir=args.out_dir, image_size=args.image_size, batch_size=args.batch_size,
        workers=args.workers, seed=args.seed, head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs, unfreeze_from=args.unfreeze_from,
        field_oversample=args.field_oversample, aug_strength=args.aug_strength,
        balance_classes=not args.no_balance, select_on=args.select_on,
        amp_dtype=args.amp_dtype, compile_model=args.compile,
    )

    D.seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- device ----
    if not torch.cuda.is_available():
        log("CUDA NOT available — this will be extremely slow on CPU.")
        log("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        device = torch.device("cpu")
        cfg.channels_last = False
        cfg.amp_dtype = "off"
    else:
        device = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        log(f"CUDA: {p.name} ({p.total_memory/1e9:.1f} GB) | torch {torch.__version__} "
            f"| cuda {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ---- data ----
    pv_root = Path(args.plantvillage_root) if args.plantvillage_root else (
        download(PLANTVILLAGE_SLUG) if args.download else
        sys.exit("Pass --plantvillage-root or --download"))
    pd_root = Path(args.plantdoc_root) if args.plantdoc_root else (
        download(PLANTDOC_SLUG) if args.download else
        sys.exit("Pass --plantdoc-root or --download"))
    warn_about_wsl_paths(pv_root)
    warn_about_wsl_paths(pd_root)

    blend = D.build_blend(pv_root, pd_root, seed=cfg.seed,
                          restrict_to_field_classes=args.restrict_to_field_classes,
                          log=log)
    classes = blend["classes"]
    class_to_idx = blend["class_to_idx"]
    splits = blend["splits"]
    num_classes = len(classes)

    if args.verify_images:
        log("Verifying image integrity (cached in the out dir)...")
        for name in list(splits):
            splits[name], _ = D.filter_unreadable(
                splits[name], out_dir / "image_cache.json", log=log)

    train_tf = D.build_train_transform(cfg.image_size, cfg.aug_strength)
    eval_tf = D.build_eval_transform(cfg.image_size)
    log(f"Train augmentation: {' -> '.join(D.describe_transform(train_tf))}")
    log(f"Eval  preprocessing: {' -> '.join(D.describe_transform(eval_tf))}")

    train_ds = D.BlendedDataset(splits["train"], class_to_idx, train_tf)
    # val_blended is kept in `splits` for size reporting only; it is the union of
    # val_studio and val_field, and evaluating it would just re-read the same
    # images a third time.
    eval_sets = {
        name: D.BlendedDataset(splits[name], class_to_idx, eval_tf)
        for name in ("val_studio", "val_field", "test_studio", "test_field")
    }

    weights = D.build_sample_weights(
        [c for _, c, _ in splits["train"]],
        [d for _, _, d in splits["train"]],
        cfg.field_oversample, cfg.balance_classes)
    share = D.effective_domain_share(weights, [d for _, _, d in splits["train"]])
    log(f"Field-domain share per epoch after oversampling x{cfg.field_oversample}: {share:.1%} "
        f"(raw share {len([1 for _,_,d in splits['train'] if d==D.DOMAIN_FIELD])/len(splits['train']):.1%})")
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)

    dl_common = dict(num_workers=cfg.workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=cfg.workers > 0,
                     prefetch_factor=4 if cfg.workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              drop_last=True, **dl_common)
    eval_loaders = {k: DataLoader(v, batch_size=cfg.batch_size * 2, shuffle=False, **dl_common)
                    for k, v in eval_sets.items()}

    label_map = {}
    for i, c in enumerate(classes):
        crop, _, cond = c.partition("___")
        label_map[str(i)] = {
            "raw_label": c,
            "crop": crop.replace("_", " ").strip(),
            "condition": (cond or "unknown").replace("_", " ").strip(),
            "healthy": cond.lower() == "healthy",
            "field_coverage": c in blend["field_classes"],
        }
    (out_dir / "labels.json").write_text(json.dumps(label_map, indent=2))

    if blend["studio_only_classes"]:
        log(f"NOTE: {len(blend['studio_only_classes'])} classes have NO field training "
            f"data; their real-world behaviour is unvalidated:")
        for c in blend["studio_only_classes"]:
            log(f"    - {c}")

    # ---- model ----
    model = build_model(num_classes).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile_model:
        log("torch.compile enabled (slow first epoch)")
        model = torch.compile(model)
    raw_model = lambda: model._orig_mod if hasattr(model, "_orig_mod") else model

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg.amp_dtype == "fp16" and device.type == "cuda"))

    history, best_score, best_epoch, patience = [], -1.0, -1, 0
    start_epoch = 0
    pending_resume = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model().load_state_dict(ck["model"])
        start_epoch = ck.get("epoch", 0) + 1
        best_score = ck.get("best_score", -1.0)
        best_epoch = ck.get("best_epoch", -1)
        history = ck.get("history", [])
        patience = ck.get("patience", 0)
        # Optimizer/scheduler/scaler state must come back too. Restoring only
        # the weights silently restarts AdamW with zeroed moments and replays
        # the LR warmup, which looks like training but is not the run you
        # thought you were continuing.
        pending_resume = {k: ck.get(k) for k in ("optimizer", "scheduler", "scaler", "phase")}
        opt_state = ("present" if pending_resume["optimizer"] else
                     "MISSING (older checkpoint; AdamW moments restart at zero)")
        log(f"Resumed from {args.resume} at epoch {start_epoch} | optimizer state: {opt_state}")
        if start_epoch >= cfg.head_epochs + cfg.finetune_epochs:
            sys.exit(f"Checkpoint is already at epoch {start_epoch} of "
                     f"{cfg.head_epochs + cfg.finetune_epochs}; nothing to resume. "
                     f"Raise --finetune-epochs to continue training.")

    total_epochs = cfg.head_epochs + cfg.finetune_epochs
    select_desc = {
        "field": f"macro-F1 on the field val split ({len(splits['val_field'])} PlantDoc images)",
        "studio": f"macro-F1 on the studio val split ({len(splits['val_studio'])} PlantVillage images)",
        "blended": (f"mean of studio and field macro-F1 "
                    f"({len(splits['val_studio'])} + {len(splits['val_field'])} images, "
                    f"weighted equally by domain, not by count)"),
    }[cfg.select_on]
    log(f"Checkpoint selection: {select_desc}")
    if cfg.select_on == "field":
        log("  (small split — expect noisy epoch-to-epoch scores; that is the "
            "price of selecting on the distribution that actually matters)")

    def make_optimizer(phase):
        if phase == 1:
            set_backbone_trainable(model, None)
            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": cfg.head_lr}]
        else:
            set_backbone_trainable(model, cfg.unfreeze_from)
            bb, hd = [], []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    (hd if "classifier" in name else bb).append(p)
            groups = [{"params": bb, "lr": cfg.backbone_lr},
                      {"params": hd, "lr": cfg.head_lr_phase2}]
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    cur_phase, optimizer, scheduler = 0, None, None

    for epoch in range(start_epoch, total_epochs):
        phase = 1 if epoch < cfg.head_epochs else 2
        if phase != cur_phase:
            cur_phase = phase
            optimizer = make_optimizer(phase)
            remaining = (cfg.head_epochs - epoch) if phase == 1 else (total_epochs - epoch)
            steps = max(1, remaining * len(train_loader))
            warm = min(cfg.warmup_epochs * len(train_loader), steps // 10)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, max(1, warm)),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps - warm)),
                ],
                milestones=[max(1, warm)])
            if pending_resume and pending_resume.get("phase") == phase:
                try:
                    if pending_resume.get("optimizer"):
                        optimizer.load_state_dict(pending_resume["optimizer"])
                    if pending_resume.get("scheduler"):
                        scheduler.load_state_dict(pending_resume["scheduler"])
                    if pending_resume.get("scaler"):
                        scaler.load_state_dict(pending_resume["scaler"])
                    log("  restored optimizer / scheduler / scaler state from checkpoint")
                except Exception as exc:  # noqa: BLE001
                    log(f"  WARNING could not restore optimizer state ({exc}); "
                        f"continuing with a fresh optimizer")
            else:
                # Fresh phase (or resumed into a different phase): patience gets
                # its own budget, because unfreezing the backbone reliably causes
                # a one- to two-epoch dip as BN stats adapt to the new domain mix.
                patience = 0
            pending_resume = None
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"--- Phase {phase} "
                f"({'head only' if phase == 1 else f'features[{cfg.unfreeze_from}:] unfrozen'}) "
                f"| {n_train:,} trainable params ---")

        unfreeze = None if phase == 1 else cfg.unfreeze_from
        desc = f"epoch {epoch+1}/{total_epochs}"
        tr_loss, tr_acc, fshare = run_epoch(model, train_loader, criterion, optimizer,
                                            scaler, scheduler, cfg, device, unfreeze, desc,
                                            args.max_steps_per_epoch)

        row = {"epoch": epoch + 1, "phase": phase, "train_loss": tr_loss,
               "train_acc": tr_acc, "train_field_share": fshare}
        for name in ("val_studio", "val_field"):
            lg, lb = collect_logits(model, eval_loaders[name], cfg, device)
            row[f"{name}_acc"] = float((lg.argmax(1) == lb).float().mean())
            row[f"{name}_macro_f1"] = macro_f1(lg, lb)
        # Mean of the two per-domain macro-F1s. Deliberately NOT the macro-F1 of
        # the pooled split: pooling lets 5,430 studio images drown out 351 field
        # ones, which is the bias this whole rewrite exists to remove.
        row["val_domain_mean_macro_f1"] = (
            row["val_studio_macro_f1"] + row["val_field_macro_f1"]) / 2

        score = {"field": row["val_field_macro_f1"],
                 "studio": row["val_studio_macro_f1"],
                 "blended": row["val_domain_mean_macro_f1"]}[cfg.select_on]

        log(f"{desc}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"STUDIO val acc {row['val_studio_acc']:.4f} F1 {row['val_studio_macro_f1']:.4f} | "
            f"FIELD val acc {row['val_field_acc']:.4f} F1 {row['val_field_macro_f1']:.4f} | "
            f"gap {row['val_studio_acc']-row['val_field_acc']:+.4f}")
        history.append(row)

        improved = score > best_score
        if improved:
            best_score, best_epoch, patience = score, epoch, 0
        else:
            patience += 1

        state = {"model": raw_model().state_dict(), "epoch": epoch,
                 "best_score": best_score, "best_epoch": best_epoch,
                 "patience": patience, "phase": phase,
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(),
                 "history": history, "classes": classes, "config": asdict(cfg)}
        torch.save(state, out_dir / "last.pt")
        if improved:
            torch.save(state, out_dir / "best.pt")
            log(f"  new best ({cfg.select_on} macroF1 {best_score:.4f}) -> best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if not improved and phase == 2 and patience >= cfg.early_stop_patience:
            log(f"Early stopping: no improvement for {cfg.early_stop_patience} epochs.")
            break

    # ---- final evaluation ----
    log("")
    if not (out_dir / "best.pt").exists():
        sys.exit(f"No best.pt in {out_dir} — no epoch completed, so there is nothing "
                 f"to evaluate or export.")
    log(f"Loading best checkpoint (epoch {best_epoch+1}, {cfg.select_on} macroF1 {best_score:.4f})")
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    raw_model().load_state_dict(best["model"])

    results, logit_cache = {}, {}
    for name in ("val_field", "test_studio", "test_field"):
        lg, lb = collect_logits(model, eval_loaders[name], cfg, device)
        logit_cache[name] = (lg, lb)
        results[name] = {
            "n": int(lb.numel()),
            "accuracy": float((lg.argmax(1) == lb).float().mean()),
            "macro_f1": macro_f1(lg, lb),
        }

    from sklearn.metrics import precision_recall_fscore_support
    for name in ("test_studio", "test_field"):
        lg, lb = logit_cache[name]
        p, r, f, _ = precision_recall_fscore_support(
            lb.numpy(), lg.argmax(1).numpy(), average="macro", zero_division=0)
        results[name].update({"macro_precision": float(p), "macro_recall": float(r)})
        write_reports(out_dir, name.replace("test_", ""), lg, lb, classes)

    with open(out_dir / "per_class_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "class", "precision", "recall", "f1", "support"])
        for name in ("test_studio", "test_field"):
            lg, lb = logit_cache[name]
            for row_ in per_class_table(lg, lb, classes):
                w.writerow([name.replace("test_", ""), row_["class"],
                            f"{row_['precision']:.4f}", f"{row_['recall']:.4f}",
                            f"{row_['f1']:.4f}", row_["support"]])

    log("")
    log("=" * 78)
    log(f"STUDIO test (PlantVillage): acc {results['test_studio']['accuracy']:.4f} "
        f"| macro P {results['test_studio']['macro_precision']:.4f} "
        f"R {results['test_studio']['macro_recall']:.4f} "
        f"F1 {results['test_studio']['macro_f1']:.4f}  (n={results['test_studio']['n']})")
    log(f"FIELD  test (PlantDoc):     acc {results['test_field']['accuracy']:.4f} "
        f"| macro P {results['test_field']['macro_precision']:.4f} "
        f"R {results['test_field']['macro_recall']:.4f} "
        f"F1 {results['test_field']['macro_f1']:.4f}  (n={results['test_field']['n']})")
    log(f"DOMAIN GAP (studio acc - field acc): "
        f"{results['test_studio']['accuracy'] - results['test_field']['accuracy']:+.4f}")
    log("The FIELD row is the one that predicts app behaviour.")
    log("=" * 78)

    # ---- calibration on FIELD data ----
    vf_lg, vf_lb = logit_cache["val_field"]
    temperature = fit_temperature(vf_lg, vf_lb)
    tf_lg, tf_lb = logit_cache["test_field"]
    rows_field = threshold_table(tf_lg, tf_lb, temperature)
    rows_studio = threshold_table(*logit_cache["test_studio"], temperature)
    recommended = pick_threshold(rows_field, cfg.target_selective_accuracy)

    log("")
    log(f"Temperature (fitted on field val, n={vf_lb.numel()}): {temperature:.4f}")
    log(f"Threshold sweep on FIELD test (n={tf_lb.numel()} — small, treat as indicative):")
    for r in rows_field:
        acc = ("  n/a " if r["selective_accuracy"] is None
               else f"{r['selective_accuracy']:.4f}")
        log(f"  t={r['threshold']:.2f} coverage {r['coverage']:.3f} "
            f"(n={r['n_accepted']:>4}) selective acc {acc}")
    log(f"Recommended confidence threshold: {recommended} "
        f"(target {cfg.target_selective_accuracy:.0%} accuracy on accepted field predictions)")

    # ---- export ----
    metadata = {
        "model_name": "mobilenetv2-agrivert-blended",
        "architecture": "mobilenet_v2",
        "datasets": {"studio": PLANTVILLAGE_SLUG, "field": PLANTDOC_SLUG},
        "num_classes": num_classes,
        "classes": classes,
        "field_covered_classes": blend["field_classes"],
        "studio_only_classes": blend["studio_only_classes"],
        "preprocessing": {
            "image_size": cfg.image_size,
            "resize_smallest_side_to": int(round(cfg.image_size * 1.14)),
            "center_crop": cfg.image_size,
            "mean": list(D.IMAGENET_MEAN), "std": list(D.IMAGENET_STD),
            "color_space": "RGB",
            "note": "Apply EXIF orientation correction before resizing.",
        },
        "calibration": {
            "temperature": temperature,
            "temperature_fitted_on": "PlantDoc validation split (field)",
            "recommended_confidence_threshold": recommended,
            "target_selective_accuracy": cfg.target_selective_accuracy,
            "threshold_sweep_field": rows_field,
            "threshold_sweep_studio": rows_studio,
            "threshold_sweep_measured_on": "held-out PlantDoc test split (field)",
        },
        "metrics": results,
        "domain_gap_accuracy": (results["test_studio"]["accuracy"]
                                - results["test_field"]["accuracy"]),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "train_split_summary": D.split_summary(splits["train"]),
        "augmentation": D.describe_transform(train_tf),
        "training_config": asdict(cfg),
        "best_epoch": best_epoch + 1,
        "caveat": (
            "Studio (PlantVillage) metrics massively overstate real-world "
            "performance; use the FIELD (PlantDoc) metrics for any decision "
            "about shipping. Classes in studio_only_classes have zero field "
            "training or validation data. Always enforce "
            "calibration.recommended_confidence_threshold at serve time and "
            "return an 'uncertain' verdict below it."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    onnx_path = out_dir / "mobilenetv2_agrivert.onnx"
    try:
        export_model = build_model(num_classes, pretrained=False)
        export_model.load_state_dict(best["model"])
        export_model.eval()
        torch.onnx.export(
            export_model, torch.randn(1, 3, cfg.image_size, cfg.image_size), str(onnx_path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17)
        log(f"Exported ONNX -> {onnx_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"ONNX export failed ({exc}); best.pt is still usable.")

    log("")
    log(f"Done. Artifacts in {out_dir.resolve()}")
    log("Serve from metadata.json — do not hardcode normalization or the threshold.")


if __name__ == "__main__":
    main()
