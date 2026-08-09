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
