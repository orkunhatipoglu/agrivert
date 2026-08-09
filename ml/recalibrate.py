#!/usr/bin/env python3
"""Re-fit the confidence threshold of a trained model, without retraining.

    python -m ml.recalibrate artifacts/vertical --target 0.85 --write

The threshold is the only thing standing between a 54%-accurate field
prediction and a grower acting on it, and the right setting is a product
decision rather than a training one: it trades how often the model answers
against how often it is right when it does. That decision should not require
a GPU and two hours to revisit.

This rebuilds the *exact* splits the training run used — same seed, same
near-duplicate grouping, same per-domain ratios — loads `best.pt`, and sweeps
the threshold finely. `--write` updates `metadata.json` in place, after which
the bundle must be re-packed so its hashes match.

Two properties this deliberately has, both of which the first version lacked:

  * The threshold is **fitted on val and verified on test**. Fitting it on the
    test split and then quoting that split's selective accuracy is circular —
    the number is what the fit maximised, so it cannot fail. On this model the
    difference was 90.8% (fitted) vs 86.4% (measured on unseen data).
  * A threshold qualifies on the **lower bound of its 95% CI**, not the point
    estimate. With ~90 accepted images a measured 91% reaches below 83%, so
    selecting on the point estimate picks whichever threshold got lucky.

Exit status is 2 when no threshold meets the target — the model is then
recorded with `meets_target: false` and serving must not advertise the target.
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


def build_splits(args):
    """Reproduce the training run's val and test splits.

    Both are needed: the threshold is fitted on val and verified on test.
    Fitting it on test and then reporting that split's selective accuracy is
    circular — the number cannot fail, because it is what the fit maximised.
    """
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
    _, val_idx, test_idx = stratified_split(
        samples,
        (0.8, 0.1, 0.1),
        args.seed,
        domain_fracs={DOMAIN_VERTICAL: (1.0 - 2 * h, h, h)},
        groups=groups,
    )
    return samples, val_idx, test_idx


def build_test_split(args):
    """Back-compat shim: the test split alone."""
    samples, _, test_idx = build_splits(args)
    return samples, test_idx


def fine_sweep(probs, targets, thresholds, min_accepted: int):
    """(threshold, coverage, n, selective accuracy + 95% CI) at each threshold.

    The CI is the point of this: with ~90 accepted images a measured 91%
    reaches below 83% at the bottom of its interval, so choosing on the point
    estimate picks whichever threshold got lucky rather than the one that will
    hold up on new photos.
    """
    from ml.train_vertical import wilson_interval

    confidence, preds = probs.max(1)
    correct = preds == targets
    rows = []
    for t in thresholds:
        accepted = confidence >= t
        n = int(accepted.sum())
        hits = int(correct[accepted].sum())
        lo, hi = wilson_interval(hits, n)
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": n / len(targets),
                "n_accepted": n,
                "selective_accuracy": (hits / n) if n else 0.0,
                "selective_accuracy_95ci": [lo, hi],
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
    ap.add_argument(
        "--policy",
        choices=("max-coverage", "conservative"),
        default="max-coverage",
        help=(
            "max-coverage: lowest gate meeting --target, i.e. the most answers "
            "at the required quality. conservative: the most selective gate "
            "still measurable on >=--min-accepted images — fewest answers, "
            "highest accuracy, for when a wrong answer costs more than none."
        ),
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

    samples, val_idx, test_idx = build_splits(args)
    pooled_val = [i for i in val_idx if samples[i].domain in domains]
    pooled_test = [i for i in test_idx if samples[i].domain in domains]
    log.info(
        "pooled %s: val %d images, test %d images",
        "+".join(domains), len(pooled_val), len(pooled_test),
    )
    if not pooled_val:
        log.error("no val images in those domains — nothing to fit on")
        return 1

    eval_tf = build_eval_transform(image_size)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            s = samples[self.indices[i]]
            return eval_tf(image=load_image_rgb(s.path))["image"], CLASS_INDEX[s.label]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(meta["architecture"], len(VERTICAL_CLASSES)).to(device)
    load_checkpoint_into(model, args.artifacts / "best.pt", map_location=device)

    def probs_for(indices):
        loader = DataLoader(
            Dataset(indices), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        )
        _, _, logits, targets = evaluate(model, loader, device, len(VERTICAL_CLASSES))
        return torch.softmax(logits / temperature, dim=1), targets

    val_probs, val_targets = probs_for(pooled_val)
    grid = np.arange(0.50, 0.996, 0.005)
    rows = fine_sweep(val_probs, val_targets, grid, args.min_accepted)

    # A gate qualifies on the CI *lower bound*, never the point estimate —
    # judging on the point estimate is what let a threshold be picked because
    # it got lucky on a handful of photos.
    reliable = [r for r in rows if r["reliable"]]
    usable = [r for r in reliable if r["selective_accuracy_95ci"][0] >= args.target]
    meets_target = bool(usable)

    print(f"\nFitted on val/{'+'.join(domains)}: {len(pooled_val)} images")
    print(f"temperature {temperature:.4f} (unchanged)\n")
    print(f"{'thresh':>7} {'coverage':>9} {'n':>5} {'accuracy':>9} {'95% CI':>15}")
    print("-" * 50)
    for r in rows:
        if round(r["threshold"] * 1000) % 25:  # print every 0.025
            continue
        flag = "" if r["reliable"] else "  (too few)"
        lo, hi = r["selective_accuracy_95ci"]
        print(
            f"{r['threshold']:>7.3f} {r['coverage']:>8.1%} {r['n_accepted']:>5} "
            f"{r['selective_accuracy']:>8.1%} {lo:>7.1%}-{hi:<7.1%}{flag}"
        )

    if not reliable:
        print(
            f"\nNo threshold accepts {args.min_accepted}+ val images. "
            "Nothing measurable here; leaving metadata unchanged."
        )
        return 1

    if args.policy == "conservative":
        # Refuse as much as we can still measure. Chosen when a wrong answer
        # costs a grower more than no answer does — most photos come back
        # "uncertain", which is the intended behaviour, not a failure.
        chosen = max(reliable, key=lambda r: r["threshold"])
    elif meets_target:
        # Lowest qualifying gate => the most answers at the required quality.
        # Anything higher only refuses more photos for no gain.
        chosen = min(usable, key=lambda r: r["threshold"])
    else:
        # Best *defensible* gate, not merely the most selective: picking the
        # highest threshold would just track --min-accepted.
        chosen = max(reliable, key=lambda r: r["selective_accuracy_95ci"][0])

    if not meets_target:
        print(
            f"\n*** No threshold reaches {args.target:.0%} selective accuracy "
            f"(95% CI lower bound) on >={args.min_accepted} val images.\n"
            f"    Using {chosen['threshold']:.3f} ({args.policy}) at "
            f"{chosen['selective_accuracy_95ci'][0]:.1%} lower bound "
            f"({chosen['selective_accuracy']:.1%} point estimate).\n"
            f"    This model does not meet the declared bar — recording "
            f"meets_target=false. Do not quote {args.target:.0%} anywhere."
        )

    # The honest number: the chosen gate measured on data it was NOT fitted on.
    verified = None
    if pooled_test:
        test_probs, test_targets = probs_for(pooled_test)
        verified = fine_sweep(
            test_probs, test_targets, [chosen["threshold"]], args.min_accepted
        )[0]
        lo, hi = verified["selective_accuracy_95ci"]
        print(
            f"\nchosen {chosen['threshold']:.3f}\n"
            f"  fitted   on val : coverage {chosen['coverage']:.1%} "
            f"(n={chosen['n_accepted']}) accuracy {chosen['selective_accuracy']:.1%}\n"
            f"  VERIFIED on test: coverage {verified['coverage']:.1%} "
            f"(n={verified['n_accepted']}) accuracy "
            f"{verified['selective_accuracy']:.1%} (95% CI {lo:.1%}-{hi:.1%})  "
            f"<- quote this one"
        )
    else:
        print(f"\nchosen {chosen['threshold']:.3f} (no test split to verify against)")

    if args.write:
        cal = meta["calibration"]
        previous = cal.get("recommended_confidence_threshold")
        cal["recommended_confidence_threshold"] = chosen["threshold"]
        cal["threshold_sweep"] = rows
        cal["threshold_fitted_on"] = f"val/{'+'.join(domains)}"
        cal["threshold_verified_on"] = f"test/{'+'.join(domains)}"
        cal["verified"] = verified
        cal["meets_target"] = meets_target
        cal["min_accepted"] = args.min_accepted
        cal["target_selective_accuracy"] = args.target
        cal["threshold_policy"] = args.policy
        cal["selection_rule"] = (
            (
                f"most selective threshold measurable on >={args.min_accepted} "
                f"val images of {'+'.join(domains)}"
            )
            if args.policy == "conservative"
            else (
                f"lowest threshold whose 95% CI lower bound on selective "
                f"accuracy reaches {args.target:.0%} on >={args.min_accepted} "
                f"val images of {'+'.join(domains)}"
            )
        )
        # Superseded by threshold_fitted_on / threshold_verified_on. Left
        # behind, it would keep asserting the threshold was measured on test.
        cal.pop("threshold_measured_on", None)
        cal["previous_threshold"] = previous
        (args.artifacts / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"\nupdated {args.artifacts / 'metadata.json'} ({previous} -> {chosen['threshold']})")
        print("re-pack the bundle so its hashes match:")
        print(f"  python -m ml.bundle pack {args.artifacts} --version <version>")
    else:
        print("\n(dry run — pass --write to update metadata.json)")
    return 0 if meets_target else 2


if __name__ == "__main__":
    raise SystemExit(main())
