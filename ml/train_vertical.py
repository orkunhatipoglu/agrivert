#!/usr/bin/env python3
"""Fine-tune a vertical-agriculture plant disease classifier.

    python -m ml.train_vertical --download --out-dir artifacts/vertical

What differs from the previous PlantVillage pipeline, and why:

1. **Scope.** 30 classes across crops that actually grow in a rack, instead
   of 38 dominated by apple/corn/grape/soybean. Capacity spent on the
   product.

2. **Three domains, not two.** studio / field / vertical. Checkpoint
   selection defaults to `vertical` — the deployment domain — because the
   domain you select on is the domain you get. The old pipeline learned this
   the hard way going from studio-only to field.

3. **The vertical split is held out honestly.** Hydroponic lettuce is ~209
   images; a random split of something that small is mostly noise, so the
   held-out portion is reported with its confidence interval and treated as
   indicative, not precise.

4. **Calibration and threshold** are fit on vertical validation and measured
   on the vertical test split, so the `uncertain` gate is tuned for the
   photos the product will actually receive.

Artifacts are written in the shared contract format (`ml.contract`), so
serving loads them without any per-run adaptation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.contract import (
    DEFAULT_CENTER_CROP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_label_map,
)
from ml.sources import (
    DOMAIN_FIELD,
    DOMAIN_STUDIO,
    DOMAIN_VERTICAL,
    EXPENSIVE_SOURCES,
    SOURCE_REFS,
    Sample,
    collect,
    ensure_extracted,
)
from ml.taxonomy import (
    CLASS_INDEX,
    UNCOVERED_TARGETS,
    VERTICAL_CLASSES,
    coverage_report,
    validate_taxonomy,
)

log = logging.getLogger("train_vertical")

# The deployment domain. Everything else is a proxy for it. Note that it is
# not used alone for selection — see --selection-domains for why.
SELECTION_DOMAIN = DOMAIN_VERTICAL


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def stratified_split(
    samples: list[Sample],
    fracs: tuple[float, ...],
    seed: int,
    domain_fracs: dict[str, tuple[float, ...]] | None = None,
    groups: list[int] | None = None,
) -> list[list[int]]:
    """Split indices per (label, domain) so every bucket keeps its mix.

    Stratifying on the pair rather than the label alone matters here: lettuce
    healthy exists in both `vertical` and nothing else, while tomato healthy
    is overwhelmingly `studio`. Stratifying on label alone would put all the
    vertical lettuce in one split by chance.

    `domain_fracs` overrides the ratio for a specific domain. The vertical
    domain is ~200 images against studio's ~22k, yet every decision that
    matters — checkpoint selection, temperature, the confidence threshold —
    is made on its holdout. A 10% slice of 200 is 20 images, which is too few
    to fit a threshold against; giving it a larger holdout buys real
    resolution and costs training images that are oversampled 12x anyway.

    `groups` assigns each sample a near-duplicate group id (see `ml.dedup`).
    A group is allocated whole, never split, so a burst of photographs of one
    plant cannot appear on both sides of the train/val boundary. Without it
    the vertical score measures recall of memorised images: it reaches ~0.96
    F1 within two epochs and the confidence threshold is then fitted against
    that fiction. Samples default to their own singleton group, which
    reproduces plain per-image stratification.
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        buckets[(s.label, s.domain)].append(i)

    rng = random.Random(seed)
    out: list[list[int]] = [[] for _ in fracs]
    for key in sorted(buckets):
        _, domain = key
        # A separate name, not a rebind of `fracs`: rebinding would leak the
        # override onto every bucket processed after the first vertical one.
        bucket_fracs = (domain_fracs or {}).get(domain, fracs)
        idx = buckets[key]

        members: dict[int, list[int]] = defaultdict(list)
        for i in idx:
            members[groups[i] if groups else i].append(i)
        order = sorted(members)
        rng.shuffle(order)

        # Largest groups first: placing a 30-image burst after the small
        # groups would blow past whichever target it lands on. Ties broken by
        # the shuffled order so the result still varies with the seed.
        rank = {gid: pos for pos, gid in enumerate(order)}
        order.sort(key=lambda g: (-len(members[g]), rank[g]))

        n = len(idx)
        targets = [n * f for f in bucket_fracs]
        counts = [0.0] * len(fracs)
        for gid in order:
            group = members[gid]
            # Whichever split is furthest below its quota takes the group, so
            # a bucket too small to fill every split still fills train first.
            k = max(range(len(fracs)), key=lambda j: targets[j] - counts[j])
            out[k] += group
            counts[k] += len(group)
    return out


def sample_weights(
    samples: list[Sample], indices: list[int], vertical_boost: float
) -> list[float]:
    """Inverse-frequency class balancing, times a domain boost.

    Two separate imbalances stack here: tomato has ~30x the images of basil,
    and `studio` has ~50x the images of `vertical`. Left alone, the model
    would become a tomato-in-a-studio detector.
    """
    label_counts = Counter(samples[i].label for i in indices)
    domain_counts = Counter(samples[i].domain for i in indices)
    log.info("  train label counts: %s", dict(label_counts.most_common()))
    log.info("  train domain counts: %s", dict(domain_counts))

    weights = []
    for i in indices:
        s = samples[i]
        w = 1.0 / label_counts[s.label]
        if s.domain == DOMAIN_VERTICAL:
            w *= vertical_boost
        weights.append(w)
    return weights


def effective_domain_share(
    samples: list[Sample], indices: list[int], weights: list[float]
) -> dict[str, float]:
    """What fraction of a sampled epoch each domain actually gets.

    Raw counts lie once a weighted sampler is involved; this is the number
    worth reasoning about when tuning --vertical-boost.
    """
    total = sum(weights) or 1.0
    share: dict[str, float] = defaultdict(float)
    for i, w in zip(indices, weights):
        share[samples[i].domain] += w / total
    return dict(share)


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------


def download_sources(names: list[str], cache: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name in names:
        kind, ref = SOURCE_REFS[name]
        if name in EXPENSIVE_SOURCES:
            log.warning("%s is expensive: %s", name, EXPENSIVE_SOURCES[name])
        log.info("downloading %s (%s:%s)…", name, kind, ref)
        if kind == "kaggle":
            import kagglehub

            roots[name] = Path(kagglehub.dataset_download(ref))
        else:
            from huggingface_hub import snapshot_download

            roots[name] = Path(
                snapshot_download(ref, repo_type="dataset", cache_dir=str(cache))
            )
    return roots


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def evaluate(model, loader, device, num_classes: int):
    """Returns (accuracy, macro_f1, logits, targets)."""
    import torch

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.float().cpu())
            all_targets.append(targets)
    if not all_logits:
        return 0.0, 0.0, torch.empty(0), torch.empty(0)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = logits.argmax(1)
    accuracy = (preds == targets).float().mean().item()

    # Macro-F1 over classes actually present, so absent classes don't drag
    # the average toward zero and hide real movement.
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return accuracy, (sum(f1s) / len(f1s) if f1s else 0.0), logits, targets


def fit_temperature(logits, targets) -> float:
    """Temperature scaling — one parameter, fit by NLL on held-out data."""
    import torch

    if logits.numel() == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def selective_report(logits, targets, temperature: float, threshold: float) -> dict:
    """Coverage and selective accuracy at one threshold, with a 95% CI."""
    import torch

    if logits.numel() == 0:
        return {"threshold": threshold, "coverage": 0.0, "n_accepted": 0,
                "selective_accuracy": 0.0, "selective_accuracy_95ci": [0.0, 0.0]}
    probs = torch.softmax(logits / temperature, dim=1)
    confidence, preds = probs.max(1)
    accepted = confidence >= threshold
    n = int(accepted.sum())
    hits = int((preds == targets)[accepted].sum())
    lo, hi = wilson_interval(hits, n)
    return {
        "threshold": round(float(threshold), 4),
        "coverage": n / len(targets),
        "n_accepted": n,
        "selective_accuracy": (hits / n) if n else 0.0,
        "selective_accuracy_95ci": [lo, hi],
    }


def threshold_sweep(
    logits,
    targets,
    temperature: float,
    target_accuracy: float,
    min_accepted: int = 40,
):
    """Confidence threshold vs coverage, and the lowest threshold we can defend.

    Two rules, both learned the hard way:

    * **Judge by the lower bound of the CI, not the point estimate.** With ~90
      accepted images, a threshold that measures 91% has a 95% interval
      reaching down past 83%. Selecting on the point estimate means picking
      whichever threshold got lucky, and the luck does not survive contact with
      new photos. Requiring the *lower* bound to clear the target is the same
      question asked honestly.
    * **Ignore thresholds accepting fewer than `min_accepted` images.** 100%
      accuracy on 6 photos is not a measurement.

    Returns `(sweep, recommended)`; `recommended` is None when nothing clears
    the bar, which is a real answer and must not be papered over.
    """
    import torch

    if logits.numel() == 0:
        return [], None

    probs = torch.softmax(logits / temperature, dim=1)
    confidence, preds = probs.max(1)
    correct = preds == targets

    sweep, recommended = [], None
    for step in range(30, 100):
        t = step / 100
        accepted = confidence >= t
        n = int(accepted.sum())
        hits = int(correct[accepted].sum())
        lo, hi = wilson_interval(hits, n)
        sweep.append(
            {
                "threshold": t,
                "coverage": n / len(targets),
                "n_accepted": n,
                "selective_accuracy": (hits / n) if n else 0.0,
                "selective_accuracy_95ci": [lo, hi],
                "reliable": n >= min_accepted,
            }
        )
        if recommended is None and n >= min_accepted and lo >= target_accuracy:
            recommended = t
    return sweep, recommended


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """95% CI for an accuracy. The vertical test split is small enough that
    a bare point estimate would be misleading."""
    if total == 0:
        return (0.0, 0.0)
    z, p, n = 1.96, correct / total, total
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/vertical"))
    ap.add_argument("--download", action="store_true", help="Fetch missing sources")
    ap.add_argument(
        "--sources",
        default="plantvillage,plantdoc,plantwild,plantseg,lettuce_hydroponic",
        help=(
            "Comma-separated. Excludes lettuce_greenhouse and combined by "
            "default — see ml.sources.EXPENSIVE_SOURCES for why."
        ),
    )
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Use an already-downloaded source, e.g. --root plantwild=/data/pw",
    )
    ap.add_argument("--architecture", default="mobilenet_v2")
    ap.add_argument("--image-size", type=int, default=DEFAULT_CENTER_CROP)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=17)
    ap.add_argument("--unfreeze-from", type=int, default=7)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--vertical-boost", type=float, default=12.0)
    ap.add_argument(
        "--vertical-holdout",
        type=float,
        default=0.2,
        help=(
            "Fraction of vertical images held out for val AND for test "
            "(0.2 => 60/20/20). Larger than the 10%% used for studio/field "
            "because every selection and calibration decision is made on it."
        ),
    )
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help=(
            "Split per image instead of per subject. The vertical source is "
            "burst photography, so this leaks near-identical frames across "
            "the split and inflates every vertical number. Diagnostic only."
        ),
    )
    ap.add_argument(
        "--dedup-threshold",
        type=int,
        default=5,
        help="Max dHash Hamming distance (of 64) counted as the same subject",
    )
    ap.add_argument("--target-selective-accuracy", type=float, default=0.90)
    ap.add_argument(
        "--min-accepted",
        type=int,
        default=40,
        help=(
            "Ignore thresholds accepting fewer than this many val images. "
            "A 99%% accuracy measured on 6 photos is not a measurement."
        ),
    )
    ap.add_argument(
        "--selection-domains",
        default=f"{DOMAIN_VERTICAL},{DOMAIN_FIELD}",
        help=(
            "Domains whose mean macro-F1 selects the checkpoint and fits the "
            "calibration. Vertical alone is ~40 held-out images covering 4 of "
            "30 classes, so it is far too noisy to early-stop on: it peaked "
            "by luck at epoch 3 and stopped a run whose field score was still "
            "climbing 18 F1 points later. Pooling with field keeps the target "
            "domain in charge without letting noise end the run."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/agrivert")
    ap.add_argument(
        "--max-steps-per-epoch", type=int, default=None, help="Smoke-test cap"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report the blend, then stop before training",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    validate_taxonomy()
    print(coverage_report())
    print()

    # ---- gather ----------------------------------------------------------
    roots: dict[str, Path] = {}
    for spec in args.root:
        name, _, path = spec.partition("=")
        roots[name] = Path(path).expanduser()

    wanted = [s for s in args.sources.split(",") if s]
    missing = [s for s in wanted if s not in roots]
    if missing:
        if not args.download:
            log.error(
                "no path given for %s. Pass --download, or --root NAME=PATH.",
                ", ".join(missing),
            )
            return 1
        roots.update(download_sources(missing, args.cache))

    # Applied to --root paths too, not just downloads: a source handed over as
    # a local path is just as likely to still be zipped.
    roots = {n: ensure_extracted(n, p, args.cache) for n, p in roots.items()}

    samples = collect(roots)
    if not samples:
        log.error("no images matched the taxonomy — check the source paths")
        return 1

    present = Counter(s.label for s in samples)
    absent = [c for c in VERTICAL_CLASSES if c not in present]
    log.info("collected %d images across %d classes", len(samples), len(present))
    if absent:
        log.warning(
            "%d taxonomy class(es) got ZERO images and cannot be learned: %s",
            len(absent),
            ", ".join(absent),
        )

    by_domain = Counter(s.domain for s in samples)
    log.info("domain totals: %s", dict(by_domain))
    if by_domain.get(DOMAIN_VERTICAL, 0) == 0:
        log.warning(
            "no %s-domain images. Checkpoint selection will fall back to "
            "field, and the result is NOT tuned for a grow rack.",
            DOMAIN_VERTICAL,
        )

    # ---- split -----------------------------------------------------------
    # ---- near-duplicate grouping ----------------------------------------
    groups = None
    if not args.no_dedup:
        from ml.dedup import compute_hashes, group_within_buckets

        hashed = compute_hashes(
            [s.path for s in samples],
            cache_file=args.cache / "dedup-hashes.json",
            workers=max(2, args.workers // 2),
        )
        groups = group_within_buckets(
            [(s.label, s.domain) for s in samples],
            [hashed.get(str(s.path)) for s in samples],
            args.dedup_threshold,
        )
        for domain in (DOMAIN_STUDIO, DOMAIN_FIELD, DOMAIN_VERTICAL):
            in_domain = [i for i, s in enumerate(samples) if s.domain == domain]
            if not in_domain:
                continue
            distinct = len({groups[i] for i in in_domain})
            log.info(
                "  %-8s %5d images -> %5d distinct subjects (%.0f%% are "
                "near-duplicates of another)",
                domain,
                len(in_domain),
                distinct,
                100 * (1 - distinct / len(in_domain)),
            )

    vertical_holdout = args.vertical_holdout
    train_idx, val_idx, test_idx = stratified_split(
        samples,
        (0.8, 0.1, 0.1),
        args.seed,
        domain_fracs={
            DOMAIN_VERTICAL: (
                1.0 - 2 * vertical_holdout,
                vertical_holdout,
                vertical_holdout,
            )
        },
        groups=groups,
    )
    log.info(
        "split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx)
    )
    for name, idx in (("val", val_idx), ("test", test_idx)):
        counts = Counter(samples[i].domain for i in idx)
        log.info("  %s by domain: %s", name, dict(counts))

    weights = sample_weights(samples, train_idx, args.vertical_boost)
    shares = effective_domain_share(samples, train_idx, weights)
    log.info(
        "effective epoch composition: %s",
        {k: f"{v:.1%}" for k, v in sorted(shares.items())},
    )

    if args.dry_run:
        print("\n--dry-run: blend verified, stopping before training.")
        return 0

    # ---- train -----------------------------------------------------------
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from ml.data import build_eval_transform, build_train_transform, load_image_rgb
    from ml.model import build_backbone, save_checkpoint

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    train_tf = build_train_transform(args.image_size)
    eval_tf = build_eval_transform(args.image_size)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            s = samples[self.indices[i]]
            image = load_image_rgb(s.path)
            return self.transform(image=image)["image"], CLASS_INDEX[s.label]

    def domain_subset(indices, domain):
        return [i for i in indices if samples[i].domain == domain]

    common = dict(
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        Dataset(train_idx, train_tf),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True
        ),
        drop_last=True,
        **common,
    )
    val_loaders = {
        d: DataLoader(
            Dataset(domain_subset(val_idx, d), eval_tf),
            batch_size=args.batch_size * 2,
            shuffle=False,
            **common,
        )
        for d in (DOMAIN_STUDIO, DOMAIN_FIELD, DOMAIN_VERTICAL)
        if domain_subset(val_idx, d)
    }
    test_loaders = {
        d: DataLoader(
            Dataset(domain_subset(test_idx, d), eval_tf),
            batch_size=args.batch_size * 2,
            shuffle=False,
            **common,
        )
        for d in (DOMAIN_STUDIO, DOMAIN_FIELD, DOMAIN_VERTICAL)
        if domain_subset(test_idx, d)
    }

    num_classes = len(VERTICAL_CLASSES)
    model = build_backbone(args.architecture, num_classes, pretrained=True).to(device)

    def set_backbone_grad(enabled: bool):
        features = getattr(model, "features", None)
        if features is None:
            return
        for i, block in enumerate(features):
            requires = enabled and i >= args.unfreeze_from
            for p in block.parameters():
                p.requires_grad = requires
            if not requires:
                # Frozen BatchNorm stays in eval mode: vertical-farm lighting
                # statistics differ sharply from studio, and letting running
                # stats drift corrupts pretrained features from epoch one.
                block.eval()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, patience = [], -1.0, -1, 0
    selection_domains = [
        d for d in args.selection_domains.split(",") if d and d in val_loaders
    ] or [DOMAIN_FIELD]
    log.info(
        "selecting checkpoints on: mean macro-F1 over %s",
        ", ".join(f"val/{d}" for d in selection_domains),
    )

    total_epochs = args.head_epochs + args.finetune_epochs
    for epoch in range(total_epochs):
        phase2 = epoch >= args.head_epochs
        if epoch == 0 or epoch == args.head_epochs:
            set_backbone_grad(phase2)
            params = [
                {
                    "params": [p for p in model.classifier.parameters()],
                    "lr": args.head_lr,
                }
            ]
            if phase2:
                params.append(
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("classifier")
                        ],
                        "lr": args.backbone_lr,
                    }
                )
            optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            # Patience resets at the phase boundary: unfreezing reliably
            # causes a 1-2 epoch dip as BN adapts, and a plateau in phase 1
            # must not spend phase 2's budget.
            patience = 0
            log.info("phase %d — backbone %s", 2 if phase2 else 1,
                     "unfrozen" if phase2 else "frozen")

        model.train()
        set_backbone_grad(phase2)
        running, seen, started = 0.0, 0, time.time()
        for step, (images, targets) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        entry = {
            "epoch": epoch,
            "phase": 2 if phase2 else 1,
            "train_loss": running / max(seen, 1),
            "seconds": round(time.time() - started, 1),
        }
        for domain, loader in val_loaders.items():
            acc, f1, _, _ = evaluate(model, loader, device, num_classes)
            entry[f"val_{domain}_accuracy"] = acc
            entry[f"val_{domain}_macro_f1"] = f1
        history.append(entry)
        log.info(
            "epoch %2d | loss %.4f | %s",
            epoch,
            entry["train_loss"],
            " | ".join(
                f"{d}: acc {entry[f'val_{d}_accuracy']:.3f} f1 "
                f"{entry[f'val_{d}_macro_f1']:.3f}"
                for d in val_loaders
            ),
        )

        parts = [entry.get(f"val_{d}_macro_f1", 0.0) for d in selection_domains]
        score = sum(parts) / len(parts)
        entry["selection_score"] = score
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            save_checkpoint(
                args.out_dir / "best.pt", model, epoch=epoch, macro_f1=score
            )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break
        # Written every epoch, not at the end, so an interrupted run keeps
        # its history.
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---- calibrate + evaluate -------------------------------------------
    from ml.model import load_checkpoint_into

    load_checkpoint_into(model, args.out_dir / "best.pt", map_location=device)
    model.to(device)

    # Calibrate on the selection domains POOLED rather than on vertical alone.
    # Vertical is ~40 held-out images covering 4 of 30 classes; a threshold
    # fitted there saw almost no hard cases, recommended 0.3, and would have
    # accepted essentially every prediction in production — including the
    # field-domain ones the model gets wrong about a third of the time.
    cal_label = "+".join(selection_domains)

    def _pooled_loader(indices):
        return DataLoader(
            Dataset(
                [i for i in indices if samples[i].domain in selection_domains],
                eval_tf,
            ),
            batch_size=args.batch_size * 2,
            shuffle=False,
            **common,
        )

    _, _, val_logits, val_targets = evaluate(
        model, _pooled_loader(val_idx), device, num_classes
    )
    temperature = fit_temperature(val_logits, val_targets)
    log.info("temperature %.4f (fit on val/%s)", temperature, cal_label)

    metrics = {}
    for domain, loader in test_loaders.items():
        acc, f1, logits, targets = evaluate(model, loader, device, num_classes)
        n = len(targets)
        lo, hi = wilson_interval(int((logits.argmax(1) == targets).sum()), n)
        metrics[f"test_{domain}"] = {
            "n": n,
            "accuracy": acc,
            "macro_f1": f1,
            "accuracy_95ci": [lo, hi],
        }
        log.info(
            "test/%s: n=%d acc=%.4f (95%% CI %.3f-%.3f) macro_f1=%.4f",
            domain, n, acc, lo, hi, f1,
        )

    # The threshold is FITTED ON VAL and VERIFIED ON TEST.
    #
    # It used to be fitted on the test split, and that split's selective
    # accuracy was then quoted as the expected quality. That is circular: the
    # threshold is chosen to maximise exactly the number being reported, so the
    # report cannot fail. Measured on this model, the gap was not academic —
    # the test-fitted 0.91 read 90.8% on the split it was fitted on and 86.4%
    # on data it had not seen, at 17.7% coverage rather than 22.3%.
    sweep, recommended = threshold_sweep(
        val_logits,
        val_targets,
        temperature,
        args.target_selective_accuracy,
        args.min_accepted,
    )
    verified = None
    if any(samples[i].domain in selection_domains for i in test_idx):
        _, _, t_logits, t_targets = evaluate(
            model, _pooled_loader(test_idx), device, num_classes
        )
        if recommended is not None:
            verified = selective_report(t_logits, t_targets, temperature, recommended)

    meets_target = recommended is not None
    if not meets_target:
        # Not a fallback so much as a refusal: no gate reaches the requested
        # quality, so pick the most selective one we can still measure and
        # record that the target was missed. Serving reads `meets_target` and
        # must not advertise an accuracy this model does not have.
        # Best *defensible* gate (highest CI lower bound), not merely the most
        # selective one — the latter just tracks --min-accepted and buys a
        # wide-interval accuracy measured on a handful of photos.
        usable = [r for r in sweep if r["reliable"]]
        recommended = (
            max(usable, key=lambda r: r["selective_accuracy_95ci"][0])["threshold"]
            if usable
            else 0.95
        )
        if any(samples[i].domain in selection_domains for i in test_idx):
            verified = selective_report(t_logits, t_targets, temperature, recommended)
        log.warning(
            "NO threshold reaches %.0f%% selective accuracy (95%% CI lower "
            "bound) on >=%d val images. Using %.2f, the most selective gate "
            "still measurable. This model does not meet the declared bar — "
            "metadata records meets_target=false; do not quote the target.",
            args.target_selective_accuracy * 100, args.min_accepted, recommended,
        )
    if verified is not None:
        log.info(
            "threshold %.2f: fitted on val, verified on test -> coverage %.1f%% "
            "(n=%d), selective accuracy %.1f%% (95%% CI %.1f-%.1f)",
            recommended, verified["coverage"] * 100, verified["n_accepted"],
            verified["selective_accuracy"] * 100,
            verified["selective_accuracy_95ci"][0] * 100,
            verified["selective_accuracy_95ci"][1] * 100,
        )

    field_classes = {
        s.label for s in samples if s.domain in (DOMAIN_FIELD, DOMAIN_VERTICAL)
    }
    metadata = {
        "model_name": "agrivert-vertical",
        "architecture": args.architecture,
        "num_classes": num_classes,
        "classes": list(VERTICAL_CLASSES),
        "field_covered_classes": sorted(field_classes),
        "vertical_covered_classes": sorted(
            {s.label for s in samples if s.domain == DOMAIN_VERTICAL}
        ),
        "uncovered_target_crops": list(UNCOVERED_TARGETS),
        "preprocessing": {
            "image_size": args.image_size,
            "center_crop": args.image_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "note": "Apply EXIF orientation correction before resizing.",
        },
        "calibration": {
            "temperature": temperature,
            "temperature_fitted_on": f"val/{cal_label}",
            "recommended_confidence_threshold": recommended,
            "target_selective_accuracy": args.target_selective_accuracy,
            "threshold_sweep": sweep,
            "threshold_fitted_on": f"val/{cal_label}",
            # The only selective-accuracy number that is a measurement rather
            # than a fitted quantity. Quote this one.
            "threshold_verified_on": f"test/{cal_label}",
            "verified": verified,
            "meets_target": meets_target,
            "min_accepted": args.min_accepted,
            "selection_rule": (
                "lowest threshold whose 95% CI lower bound on selective "
                f"accuracy reaches {args.target_selective_accuracy:.0%} on "
                f">={args.min_accepted} val images of {cal_label}"
            ),
            "selection_domains": selection_domains,
        },
        "metrics": metrics,
        "best_epoch": best_epoch,
        "training_config": {k: str(v) for k, v in vars(args).items()},
        "sources": sorted(roots),
        "domain_totals": dict(Counter(s.domain for s in samples)),
        "caveat": (
            "Selected and calibrated on the VERTICAL domain (hydroponic / "
            "greenhouse). studio metrics overstate real performance badly. "
            f"These crops have NO training data at all and will be "
            f"misclassified: {', '.join(UNCOVERED_TARGETS)}. Always enforce "
            "calibration.recommended_confidence_threshold at serve time."
        ),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "labels.json").write_text(
        json.dumps(build_label_map(list(VERTICAL_CLASSES), field_classes), indent=2)
    )
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nwrote artifacts to {args.out_dir}")
    print(
        f"  best epoch {best_epoch}, mean val macro-F1 over "
        f"{cal_label} = {best_score:.4f}"
    )
    print(f"  threshold {recommended}, temperature {temperature:.4f}")
    print("\nPackage for distribution:")
    print(
        f"  python -m ml.bundle pack {args.out_dir} "
        f"--version v2-vertical-$(date +%Y%m%d)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
