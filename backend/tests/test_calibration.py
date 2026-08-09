"""Guards on how the confidence threshold is chosen.

The threshold decides whether a farmer sees a diagnosis or an honest "I can't
tell", so the number attached to it has to be a measurement rather than a
wish. Two failures are specifically pinned here because both shipped:

  * The threshold was fitted on the *test* split and that split's selective
    accuracy was then quoted as the expected quality. That is circular — the
    fit maximises exactly the number being reported, so it cannot fail. The
    shipped 0.91 read 90.8% on the split it was fitted on and 86.4% on data it
    had not seen.
  * Thresholds qualified on the point estimate. With a few dozen accepted
    images that rewards whichever threshold got lucky, and the luck does not
    survive contact with new photos.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")


def _logits(rows):
    """rows: list of (confidence_for_class_0, is_correct).

    Builds 2-class logits whose softmax max is ~the requested confidence, with
    the argmax landing on class 0 and the target set so the prediction is
    right or wrong as asked.
    """
    lg, tg = [], []
    for conf, correct in rows:
        # softmax([x, 0])[0] = conf  =>  x = log(conf / (1 - conf))
        x = math.log(conf / (1 - conf))
        lg.append([x, 0.0])
        tg.append(0 if correct else 1)
    return torch.tensor(lg), torch.tensor(tg)


def test_a_lucky_small_sample_cannot_win_the_threshold():
    """High point estimate on few images must lose to the CI lower bound."""
    from ml.train_vertical import threshold_sweep

    # 8 images above 0.95, all correct => point estimate 100%, but the 95% CI
    # reaches down near 0.63, so it must not qualify at a 90% target.
    rows = [(0.97, True)] * 8
    # 200 more between 0.5 and 0.9 that are only 70% correct.
    rows += [(0.80, i % 10 < 7) for i in range(200)]
    logits, targets = _logits(rows)

    sweep, recommended = threshold_sweep(
        logits, targets, temperature=1.0, target_accuracy=0.90, min_accepted=40
    )
    high = [r for r in sweep if r["threshold"] >= 0.95][0]
    assert high["selective_accuracy"] == 1.0, "setup: point estimate should be perfect"
    assert not high["reliable"], "8 images must be marked unreliable"
    assert recommended is None or recommended < 0.95


def test_returns_none_rather_than_inventing_a_threshold():
    """Nothing clearing the bar is a real answer, not something to paper over."""
    from ml.train_vertical import threshold_sweep

    # Uniformly mediocre: no gate can reach 90%.
    logits, targets = _logits([(0.85, i % 10 < 6) for i in range(300)])
    _, recommended = threshold_sweep(
        logits, targets, temperature=1.0, target_accuracy=0.90, min_accepted=40
    )
    assert recommended is None


def test_min_accepted_is_enforced():
    from ml.train_vertical import threshold_sweep

    rows = [(0.99, True)] * 5 + [(0.60, i % 2 == 0) for i in range(100)]
    logits, targets = _logits(rows)
    sweep, _ = threshold_sweep(
        logits, targets, temperature=1.0, target_accuracy=0.5, min_accepted=40
    )
    tiny = [r for r in sweep if r["n_accepted"] < 40]
    assert tiny and all(not r["reliable"] for r in tiny)


def test_every_sweep_row_carries_a_confidence_interval():
    """A selective accuracy without an interval is what caused the overclaim."""
    from ml.train_vertical import threshold_sweep

    logits, targets = _logits([(0.75, i % 4 != 0) for i in range(120)])
    sweep, _ = threshold_sweep(
        logits, targets, temperature=1.0, target_accuracy=0.90, min_accepted=40
    )
    for row in sweep:
        lo, hi = row["selective_accuracy_95ci"]
        assert 0.0 <= lo <= row["selective_accuracy"] <= hi <= 1.0


def test_selective_report_measures_the_gate_it_is_given():
    from ml.train_vertical import selective_report

    # 60 images at 0.9 (all correct), 40 at 0.6 (all wrong).
    logits, targets = _logits([(0.9, True)] * 60 + [(0.6, False)] * 40)
    r = selective_report(logits, targets, temperature=1.0, threshold=0.8)
    assert r["n_accepted"] == 60
    assert r["coverage"] == pytest.approx(0.6)
    assert r["selective_accuracy"] == pytest.approx(1.0)


def test_an_easy_domain_cannot_carry_a_hard_one_through_the_gate():
    """The v3 regression: pooling let 84% vertical hide field's real accuracy.

    Pooled selective accuracy at threshold 0.30 read 95.3% while a field photo
    accepted at that same gate was right 74% of the time — because the pool
    was 2268 easy vertical images against 447 field ones. Selection must be
    judged on the worst domain, not the average.
    """
    from ml.recalibrate import per_domain_sweep

    # 400 easy images, ~all correct; 100 hard ones, ~60% correct.
    easy = _logits([(0.97, True)] * 400)
    hard = _logits([(0.97, i % 10 < 6) for i in range(100)])
    grid = [0.5]
    rows = per_domain_sweep(
        {"vertical": (torch.softmax(easy[0], 1), easy[1]),
         "field": (torch.softmax(hard[0], 1), hard[1])},
        grid, min_accepted=40,
    )
    row = rows[0]
    assert row["worst_domain"] == "field"
    # Pooled would be ~(400 + 60)/500 = 92%; the worst domain is ~60%.
    assert row["per_domain"]["vertical"]["selective_accuracy"] == pytest.approx(1.0)
    assert row["per_domain"]["field"]["selective_accuracy"] == pytest.approx(0.6)
    assert row["worst_lower_bound"] < 0.75, (
        "selection score followed the pooled mean instead of the worst domain"
    )


def test_worst_domain_rule_needs_every_domain_measurable():
    """A domain with too few accepted images must make the row unreliable."""
    from ml.recalibrate import per_domain_sweep

    big = _logits([(0.97, True)] * 200)
    tiny = _logits([(0.97, True)] * 5)
    rows = per_domain_sweep(
        {"vertical": (torch.softmax(big[0], 1), big[1]),
         "field": (torch.softmax(tiny[0], 1), tiny[1])},
        [0.5], min_accepted=40,
    )
    assert not rows[0]["reliable"]


def test_recalibrate_takes_the_split_from_the_model_it_is_recalibrating():
    """Otherwise the threshold gets fitted against a split the model never had.

    Recalibration used to rely on CLI defaults matching whatever the training
    run happened to use. Add a data source and the defaults are stale: the
    sweep runs on a different held-out set, succeeds, and returns a threshold
    fitted partly against images the model trained on. No error, and the
    number looks perfectly reasonable.
    """
    import argparse

    from ml.recalibrate import adopt_split_config

    meta = {"split_config": {
        "sources": ["plantvillage", "lettuce_roboflow"],
        "roots": {"plantvillage": "/data/pv", "lettuce_roboflow": "/data/rf"},
        "seed": 7,
        "fracs": [0.8, 0.1, 0.1],
        "vertical_holdout": 0.15,
        "dedup_threshold": 9,
        "no_dedup": False,
        "selection_domains": ["vertical", "field"],
    }}
    args = argparse.Namespace(
        artifacts=Path("/tmp/x"), sources=None, root=[], seed=None,
        vertical_holdout=None, dedup_threshold=None, no_dedup=False,
        selection_domains=None,
    )
    adopt_split_config(args, meta)

    assert args.sources == "plantvillage,lettuce_roboflow"
    assert sorted(args.root) == ["lettuce_roboflow=/data/rf", "plantvillage=/data/pv"]
    assert args.seed == 7
    assert args.vertical_holdout == 0.15
    assert args.dedup_threshold == 9


def test_explicit_flags_still_beat_the_recorded_config():
    """Deviating on purpose must stay possible."""
    import argparse

    from ml.recalibrate import adopt_split_config

    meta = {"split_config": {
        "sources": ["plantvillage"], "roots": {"plantvillage": "/data/pv"},
        "seed": 7, "fracs": [0.8, 0.1, 0.1], "vertical_holdout": 0.15,
        "dedup_threshold": 9, "no_dedup": False,
        "selection_domains": ["vertical"],
    }}
    args = argparse.Namespace(
        artifacts=Path("/tmp/x"), sources="plantdoc", root=["plantdoc=/other"],
        seed=1, vertical_holdout=0.3, dedup_threshold=2, no_dedup=True,
        selection_domains="field",
    )
    adopt_split_config(args, meta)

    assert args.sources == "plantdoc"
    assert args.root == ["plantdoc=/other"]
    assert args.seed == 1 and args.dedup_threshold == 2
    assert args.selection_domains == "field"


def test_recalibrate_fits_on_val_not_test():
    """Regression: build_splits must expose val, and the fit must use it.

    The bug this pins is not a crash — it is a number. If the threshold is
    fitted on the same split it is reported on, the report is guaranteed to
    look good, which is precisely why it went unnoticed.
    """
    import inspect

    from ml import recalibrate

    src = inspect.getsource(recalibrate.main)
    assert "probs_for(pooled_val)" in src, "threshold must be fitted on val"
    fit_at = src.index("probs_for(pooled_val)")
    # The test split may only be touched *after* the threshold is chosen.
    assert "chosen" in src[:src.index("probs_for(pooled_test)")][fit_at:]
    assert "build_splits" in src
