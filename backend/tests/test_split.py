"""Guards on how the train/val/test split is formed.

Two properties matter here and neither is visible from a training log:

  * A near-duplicate group must land wholly on one side. The vertical source
    is burst photography, and splitting a burst across train/val turns the
    vertical score — the number every selection and calibration decision uses
    — into a memorisation check that reads ~0.96 F1 within two epochs.
  * A per-domain ratio override must apply to that domain only. The first
    version of this rebound the loop variable, which leaked the vertical
    ratio onto every bucket processed afterwards and silently reshaped the
    studio and field splits.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _samples(spec):
    """spec: list of (label, domain, count)."""
    from ml.sources import Sample

    out = []
    for label, domain, count in spec:
        for i in range(count):
            out.append(Sample(Path(f"/tmp/{label}_{domain}_{i}.jpg"), label, domain, "t"))
    return out


def test_a_duplicate_group_is_never_split_across_sides():
    from ml.train_vertical import stratified_split

    samples = _samples([("L___healthy", "vertical", 40)])
    # First 12 images are one burst of the same plant; the rest are distinct.
    groups = [0] * 12 + list(range(1, 29))

    splits = stratified_split(samples, (0.6, 0.2, 0.2), seed=1, groups=groups)

    burst = set(range(12))
    landed = [i for i, s in enumerate(splits) if burst & set(s)]
    assert len(landed) == 1, "the burst was spread across multiple splits"
    assert burst <= set(splits[landed[0]])


def test_every_sample_lands_exactly_once():
    from ml.train_vertical import stratified_split

    samples = _samples([("A", "studio", 50), ("B", "field", 30)])
    splits = stratified_split(samples, (0.8, 0.1, 0.1), seed=0)

    flat = [i for s in splits for i in s]
    assert sorted(flat) == list(range(len(samples)))


def test_domain_override_does_not_leak_to_other_domains():
    """The shadowing regression: a vertical override must not reshape studio."""
    from ml.train_vertical import stratified_split

    samples = _samples([("A", "studio", 100), ("L", "vertical", 100)])
    splits = stratified_split(
        samples,
        (0.8, 0.1, 0.1),
        seed=0,
        domain_fracs={"vertical": (0.6, 0.2, 0.2)},
    )

    def domain_counts(split):
        return Counter(samples[i].domain for i in split)

    train, val, test = (domain_counts(s) for s in splits)
    # studio keeps 80/10/10 regardless of the vertical override
    assert abs(train["studio"] - 80) <= 2, train
    assert abs(val["studio"] - 10) <= 2, val
    assert abs(test["studio"] - 10) <= 2, test
    # vertical follows its own 60/20/20
    assert abs(train["vertical"] - 60) <= 2, train
    assert abs(val["vertical"] - 20) <= 2, val


def test_bucket_too_small_to_fill_every_split_fills_train_first():
    """A 2-image class must still contribute to training rather than vanish."""
    from ml.train_vertical import stratified_split

    samples = _samples([("Rare", "field", 2)])
    train, _, _ = stratified_split(samples, (0.8, 0.1, 0.1), seed=0)
    assert len(train) >= 1


def test_dedup_groups_near_identical_images(tmp_path):
    """dHash must merge a re-encoded near-twin and keep a different image apart."""
    from PIL import Image

    from ml.dedup import dhash, group_duplicates

    base = Image.new("L", (64, 64))
    for x in range(64):
        for y in range(64):
            base.putpixel((x, y), (x * 4) % 256)
    a = tmp_path / "a.png"
    base.save(a)

    # Same subject, mild recompression/scale — the burst-frame case.
    b = tmp_path / "b.jpg"
    base.resize((60, 60)).resize((64, 64)).save(b, quality=80)

    other = Image.new("L", (64, 64))
    for x in range(64):
        for y in range(64):
            other.putpixel((x, y), (y * 7 + x) % 256)
    c = tmp_path / "c.png"
    other.save(c)

    groups = group_duplicates([dhash(a), dhash(b), dhash(c)], threshold=5)
    assert groups[0] == groups[1], "near-identical images were not grouped"
    assert groups[2] != groups[0], "distinct images were merged"


def test_unreadable_image_gets_its_own_group():
    """A file that cannot be hashed must not be dropped or merged wholesale."""
    from ml.dedup import group_duplicates

    groups = group_duplicates([None, None, 12345], threshold=5)
    assert len(set(groups)) == 3
