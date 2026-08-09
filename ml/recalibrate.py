#!/usr/bin/env python3
"""Re-fit the confidence threshold of a trained model, without retraining.

    python -m ml.recalibrate artifacts/vertical --target 0.85 --write

The threshold is the only thing standing between a 54%-accurate field
prediction and a grower acting on it, and the right setting is a product
decision rather than a training one: it trades how often the model answers
against how often it is right when it does. That decision should not require
a GPU and two hours to revisit.

This rebuilds the *exact* held-out split the training run used — same seed,
same near-duplicate grouping, same per-domain ratios — loads `best.pt`, and
sweeps the threshold finely over the pooled selection domains. `--write`
updates `metadata.json` in place, after which the bundle must be re-packed
so its hashes match.

The training run's own sweep is a coarse 8-point grid, which left a cliff
between 0.90 (25% coverage) and 0.95 (15%). This resolves that region.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ml.sources import DOMAIN_VERTICAL, collect, ensure_extracted
from ml.taxonomy import CLASS_INDEX, VERTICAL_CLASSES
from ml.train_vertical import download_sources, evaluate, stratified_split

log = logging.getLogger("recalibrate")


def build_test_split(args):
    """Reproduce the training run's pooled held-out set."""
    roots: dict[str, Path] = {}
    for spec in args.root:
        name, _, path = spec.partition("=")
        roots[name] = Path(path).expanduser()

    wanted = [s for s in args.sources.split(",") if s]
    missing = [s for s in wanted if s not in roots]
    if missing:
        roots.update(download_sources(missing, args.cache))
    roots = {n: ensure_extracted(n, p, args.cache) for n, p in roots.items()}

    samples = collect(roots)

    groups = None
    if not args.no_dedup:
        from ml.dedup import compute_hashes, group_within_buckets

        hashed = compute_hashes(
            [s.path for s in samples],
            cache_file=args.cache / "dedup-hashes.json",
            workers=args.workers,
        )
        groups = group_within_buckets(
            [(s.label, s.domain) for s in samples],
            [hashed.get(str(s.path)) for s in samples],
            args.dedup_threshold,
        )

    h = args.vertical_holdout
    _, _, test_idx = stratified_split(
        samples,
        (0.8, 0.1, 0.1),
        args.seed,
        domain_fracs={DOMAIN_VERTICAL: (1.0 - 2 * h, h, h)},
        groups=groups,
    )
    return samples, test_idx


def fine_sweep(probs, targets, thresholds, min_accepted: int):
    """(threshold, coverage, n, selective accuracy) at each threshold."""
    confidence, preds = probs.max(1)
    correct = preds == targets
    rows = []
    for t in thresholds:
        accepted = confidence >= t
        n = int(accepted.sum())
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": n / len(targets),
                "n_accepted": n,
                "selective_accuracy": (
                    float(correct[accepted].float().mean()) if n else 0.0
                ),
                "reliable": n >= min_accepted,
            }
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", type=Path)
    ap.add_argument(
        "--target",
        type=float,
        default=0.85,
        help="Required accuracy among ACCEPTED predictions",
    )
    ap.add_argument(
        "--min-accepted",
        type=int,
        default=40,
        help="Ignore thresholds accepting fewer than this many test images; "
        "a 99%% accuracy measured on 6 photos is not a measurement",
    )
    ap.add_argument("--selection-domains", default="vertical,field")
    ap.add_argument(
        "--sources",
        default="plantvillage,plantdoc,plantwild,plantseg,lettuce_hydroponic",
    )
    ap.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vertical-holdout", type=float, default=0.2)
    ap.add_argument("--dedup-threshold", type=int, default=5)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/agrivert")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Update metadata.json in place (re-pack the bundle afterwards)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    import torch
    from torch.utils.data import DataLoader

    from ml.data import build_eval_transform, load_image_rgb
    from ml.model import build_backbone, load_checkpoint_into

    meta = json.loads((args.artifacts / "metadata.json").read_text())
    image_size = meta["preprocessing"]["image_size"]
    temperature = meta["calibration"]["temperature"]
    domains = [d for d in args.selection_domains.split(",") if d]

    samples, test_idx = build_test_split(args)
    pooled = [i for i in test_idx if samples[i].domain in domains]
    log.info("pooled held-out set (%s): %d images", "+".join(domains), len(pooled))
    if not pooled:
        log.error("no held-out images in those domains")
        return 1

    eval_tf = build_eval_transform(image_size)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(pooled)

        def __getitem__(self, i):
            s = samples[pooled[i]]
            return eval_tf(image=load_image_rgb(s.path))["image"], CLASS_INDEX[s.label]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(meta["architecture"], len(VERTICAL_CLASSES)).to(device)
    load_checkpoint_into(model, args.artifacts / "best.pt", map_location=device)

    loader = DataLoader(
        Dataset(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    _, _, logits, targets = evaluate(model, loader, device, len(VERTICAL_CLASSES))
    probs = torch.softmax(logits / temperature, dim=1)

    rows = fine_sweep(
        probs, targets, np.arange(0.50, 0.996, 0.005), args.min_accepted
    )

    # Lowest threshold clearing the accuracy bar => the most answers we can
    # give at that quality. Anything higher only refuses more photos.
    usable = [r for r in rows if r["selective_accuracy"] >= args.target and r["reliable"]]
    chosen = min(usable, key=lambda r: r["threshold"]) if usable else None

    print(f"\nPooled {'+'.join(domains)} held-out set: {len(pooled)} images")
    print(f"temperature {temperature:.4f} (unchanged)\n")
    print(f"{'thresh':>7} {'coverage':>9} {'n':>5} {'accuracy':>9}")
    print("-" * 34)
    for r in rows:
        if round(r["threshold"] * 1000) % 25:  # print every 0.025
            continue
        flag = "" if r["reliable"] else "  (too few)"
        print(
            f"{r['threshold']:>7.3f} {r['coverage']:>8.1%} {r['n_accepted']:>5} "
            f"{r['selective_accuracy']:>8.1%}{flag}"
        )

    if chosen is None:
        print(
            f"\nNo threshold reaches {args.target:.0%} accuracy on at least "
            f"{args.min_accepted} images. Leaving metadata unchanged."
        )
        return 1

    print(
        f"\nchosen {chosen['threshold']:.3f} -> answers "
        f"{chosen['coverage']:.1%} of photos at {chosen['selective_accuracy']:.1%} "
        f"accuracy (n={chosen['n_accepted']})"
    )

    if args.write:
        cal = meta["calibration"]
        previous = cal.get("recommended_confidence_threshold")
        cal["recommended_confidence_threshold"] = chosen["threshold"]
        cal["threshold_sweep"] = rows
        cal["threshold_policy"] = (
            f"lowest threshold reaching {args.target:.0%} selective accuracy on "
            f">={args.min_accepted} held-out images of {'+'.join(domains)}"
        )
        cal["previous_threshold"] = previous
        (args.artifacts / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"updated {args.artifacts / 'metadata.json'} ({previous} -> {chosen['threshold']})")
        print("re-pack the bundle so its hashes match:")
        print(f"  python -m ml.bundle pack {args.artifacts} --version <version>")
    else:
        print("\n(dry run — pass --write to update metadata.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
