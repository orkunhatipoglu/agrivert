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


def adopt_split_config(args, meta) -> None:
    """Take the split settings from the model's own metadata.

    The alternative is re-typing the source list and every split flag from
    memory on each recalibration. That fails silently and expensively: a
    missing source or a different seed produces a *different* held-out set,
    the sweep runs happily on it, and the threshold that comes out was fitted
    against data the model may well have trained on. Nothing errors, and the
    resulting number looks entirely reasonable.

    Explicit flags still win, so a deliberate experiment is still possible.
    """
    cfg = meta.get("split_config")
    if not cfg:
        log.warning(
            "%s predates split_config; falling back to CLI defaults. Verify "
            "--sources/--root/--seed match the training run, or the threshold "
            "will be fitted on the wrong split.",
            args.artifacts,
        )
        return
    if not args.sources:
        args.sources = ",".join(cfg["sources"])
    if not args.root:
        args.root = [f"{n}={p}" for n, p in cfg["roots"].items()]
    for flag, key in (
        ("seed", "seed"),
        ("vertical_holdout", "vertical_holdout"),
        ("dedup_threshold", "dedup_threshold"),
    ):
        if getattr(args, flag) is None:
            setattr(args, flag, cfg[key])
    if not args.no_dedup:
        args.no_dedup = cfg["no_dedup"]
    if not args.selection_domains:
        args.selection_domains = ",".join(cfg["selection_domains"])
    log.info(
        "split config adopted from metadata: %d source(s), seed=%s, "
        "vertical_holdout=%s, dedup_threshold=%s",
        len(cfg["sources"]), args.seed, args.vertical_holdout, args.dedup_threshold,
    )


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


def per_domain_sweep(by_domain: dict, thresholds, min_accepted: int) -> list[dict]:
    """Sweep each selection domain separately and judge by the WORST one.

    A pooled sweep answers "how good is this gate on average over the held-out
    set", which is the question you want only while the domains are comparably
    sized. They are not. After the Roboflow import the pool is 84% vertical —
    single-session greenhouse photos the model gets 99% right — and 16% field.
    Pooled selective accuracy at threshold 0.30 reads 95.3%; field accuracy at
    that same gate is 74.0%. The average hid the domain a grower's photo
    actually resembles.

    A threshold therefore qualifies only when EVERY selection domain clears
    the bar on its own CI lower bound, with enough accepted images in each to
    measure. The gate has to hold where it is weakest, not on average.

    `by_domain` maps domain -> (probs, targets). Each returned row carries the
    per-domain detail plus `worst_lower_bound`, the value selection uses.
    """
    per = {d: {r["threshold"]: r for r in fine_sweep(p, t, thresholds, min_accepted)}
           for d, (p, t) in by_domain.items()}
    rows = []
    for t in thresholds:
        key = round(float(t), 4)
        doms = {d: per[d][key] for d in by_domain}
        reliable = all(r["reliable"] for r in doms.values())
        worst_lo = min(r["selective_accuracy_95ci"][0] for r in doms.values())
        worst_dom = min(doms, key=lambda d: doms[d]["selective_accuracy_95ci"][0])
        total = sum(r["n_accepted"] for r in doms.values())
        seen = sum(round(r["n_accepted"] / r["coverage"]) if r["coverage"] else 0
                   for r in doms.values())
        rows.append({
            "threshold": key,
            "coverage": (total / seen) if seen else 0.0,
            "n_accepted": total,
            "reliable": reliable,
            "worst_lower_bound": worst_lo,
            "worst_domain": worst_dom,
            "per_domain": {d: {
                "coverage": r["coverage"],
                "n_accepted": r["n_accepted"],
                "selective_accuracy": r["selective_accuracy"],
                "selective_accuracy_95ci": r["selective_accuracy_95ci"],
                "reliable": r["reliable"],
            } for d, r in doms.items()},
        })
    return rows


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
        "--gate-rule",
        choices=("worst-domain", "pooled"),
        default="worst-domain",
        help=(
            "worst-domain (default): a gate must clear --target in EVERY "
            "selection domain on its own. pooled: judge the domains averaged "
            "together — only meaningful while they are comparably sized, and "
            "they are not (vertical outnumbers field ~5:1), so pooled lets an "
            "easy domain carry a hard one."
        ),
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
    # These all default to "take it from the model's metadata" rather than to
    # a hardcoded value, so recalibrating cannot quietly use a split the model
    # was never trained against. Pass them only to deviate on purpose.
    ap.add_argument("--selection-domains", default=None)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vertical-holdout", type=float, default=None)
    ap.add_argument("--dedup-threshold", type=int, default=None)
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
    adopt_split_config(args, meta)
    image_size = meta["preprocessing"]["image_size"]
    temperature = meta["calibration"]["temperature"]
    domains = [d for d in (args.selection_domains or "vertical,field").split(",") if d]
    # Post-adoption fallbacks, for artifacts predating split_config.
    args.seed = 42 if args.seed is None else args.seed
    args.vertical_holdout = 0.2 if args.vertical_holdout is None else args.vertical_holdout
    args.dedup_threshold = 5 if args.dedup_threshold is None else args.dedup_threshold
    args.sources = args.sources or "plantvillage,plantdoc,plantwild,plantseg,lettuce_hydroponic"

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

    grid = np.arange(0.30, 0.996, 0.005)
    if args.gate_rule == "worst-domain":
        by_domain = {}
        for dom in domains:
            idx = [i for i in val_idx if samples[i].domain == dom]
            if not idx:
                log.warning("no val images in domain %r — it cannot gate", dom)
                continue
            by_domain[dom] = probs_for(idx)
        rows = per_domain_sweep(by_domain, grid, args.min_accepted)
        score = lambda r: r["worst_lower_bound"]  # noqa: E731
    else:
        val_probs, val_targets = probs_for(pooled_val)
        rows = fine_sweep(val_probs, val_targets, grid, args.min_accepted)
        score = lambda r: r["selective_accuracy_95ci"][0]  # noqa: E731

    # A gate qualifies on the CI *lower bound*, never the point estimate —
    # judging on the point estimate is what let a threshold be picked because
    # it got lucky on a handful of photos.
    reliable = [r for r in rows if r["reliable"]]
    usable = [r for r in reliable if score(r) >= args.target]
    meets_target = bool(usable)

    print(f"\nFitted on val/{'+'.join(domains)}  (rule: {args.gate_rule})")
    print(f"temperature {temperature:.4f} (unchanged)")
    if args.gate_rule == "worst-domain":
        counts = {d: sum(1 for i in val_idx if samples[i].domain == d) for d in domains}
        print(f"val composition: {counts}")
        print("selection uses the WORST domain's CI lower bound, not the pooled mean\n")
        head = f"{'thresh':>7}" + "".join(f"{d[:9]:>26}" for d in domains) + f"{'worst LB':>10}"
        print(head)
        print("-" * len(head))
        for r in rows:
            if round(r["threshold"] * 1000) % 50:  # every 0.05
                continue
            cells = ""
            for d in domains:
                pd = r["per_domain"].get(d)
                cells += (
                    f"{pd['coverage']:>9.0%}/{pd['n_accepted']:<5}{pd['selective_accuracy']:>7.1%}"
                    f"{'' if pd['reliable'] else '*':>5}"
                ) if pd else f"{'-':>26}"
            print(f"{r['threshold']:>7.3f}{cells}{score(r):>10.1%}")
        print("  (columns are coverage/n and selective accuracy; * = too few to measure)")
    else:
        print()
        print(f"{'thresh':>7} {'coverage':>9} {'n':>5} {'accuracy':>9} {'95% CI':>15}")
        print("-" * 50)
        for r in rows:
            if round(r["threshold"] * 1000) % 25:
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
        chosen = max(reliable, key=score)

    if not meets_target:
        worst = (
            f" (worst domain: {chosen['worst_domain']})"
            if args.gate_rule == "worst-domain" else ""
        )
        print(
            f"\n*** No threshold reaches {args.target:.0%} selective accuracy "
            f"(95% CI lower bound) on >={args.min_accepted} val images "
            f"in EVERY selection domain.\n"
            f"    Using {chosen['threshold']:.3f} ({args.policy}) at "
            f"{score(chosen):.1%} lower bound{worst}.\n"
            f"    This model does not meet the declared bar — recording "
            f"meets_target=false. Do not quote {args.target:.0%} anywhere."
        )

    # The honest number: the chosen gate measured on data it was NOT fitted on,
    # broken out per domain. A single pooled figure here would reintroduce
    # exactly the averaging that hid field behind vertical.
    verified = None
    verified_per_domain = {}
    if pooled_test:
        test_probs, test_targets = probs_for(pooled_test)
        verified = fine_sweep(
            test_probs, test_targets, [chosen["threshold"]], args.min_accepted
        )[0]
        for dom in domains:
            idx = [i for i in test_idx if samples[i].domain == dom]
            if not idx:
                continue
            p, t = probs_for(idx)
            verified_per_domain[dom] = fine_sweep(
                p, t, [chosen["threshold"]], args.min_accepted
            )[0]

        print(f"\nchosen {chosen['threshold']:.3f}  — VERIFIED on test (never fitted on)")
        print(f"{'domain':<12} {'coverage':>9} {'n':>6} {'accuracy':>9} {'95% CI':>16}")
        print("-" * 56)
        for dom, r in verified_per_domain.items():
            lo, hi = r["selective_accuracy_95ci"]
            print(f"{dom:<12} {r['coverage']:>8.1%} {r['n_accepted']:>6} "
                  f"{r['selective_accuracy']:>8.1%} {lo:>7.1%}-{hi:<7.1%}")
        lo, hi = verified["selective_accuracy_95ci"]
        print(f"{'pooled':<12} {verified['coverage']:>8.1%} {verified['n_accepted']:>6} "
              f"{verified['selective_accuracy']:>8.1%} {lo:>7.1%}-{hi:<7.1%}")
        if verified_per_domain:
            w = min(verified_per_domain,
                    key=lambda d: verified_per_domain[d]["selective_accuracy"])
            print(
                f"\nQuote the per-domain rows, not 'pooled': the pool is "
                f"{max((verified_per_domain[d]['n_accepted'] for d in verified_per_domain))} "
                f"vs {min((verified_per_domain[d]['n_accepted'] for d in verified_per_domain))} "
                f"images across domains, so its mean is dominated by the larger one. "
                f"A {w} photo accepted at this gate is right "
                f"{verified_per_domain[w]['selective_accuracy']:.1%} of the time."
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
        cal["verified_per_domain"] = verified_per_domain
        cal["gate_rule"] = args.gate_rule
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
