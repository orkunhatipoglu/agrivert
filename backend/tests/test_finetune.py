"""Guards on fine-tuning onto an expanded taxonomy.

Adding a class is the routine way this project grows, and there are two ways
to do it that look identical in the logs and are not:

  * Warm-starting the classifier head **by position**. Insert a class anywhere
    but the end and every later class silently inherits its neighbour's
    learned weights. Training still runs, loss still falls, and the only
    symptom is a model that starts worse than it should for reasons nothing
    reports. `Lettuce___Sclerotinia_rot` was inserted in the middle of the
    lettuce block precisely so this stays exercised.
  * Treating a growth stage as a diagnosis. The Roboflow export's `growing`
    and `raising_seeding` columns co-occur with `health`; mapping them as
    labels would teach the model to answer "seedling" when asked what is
    wrong with a plant.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")


def _tiny_checkpoint(tmp_path, classes):
    from ml.model import build_backbone, save_checkpoint

    model = build_backbone("mobilenet_v2", len(classes))
    # Make each head row trivially identifiable by its class index.
    with torch.no_grad():
        model.classifier[1].weight.fill_(0.0)
        model.classifier[1].bias.fill_(0.0)
        for i in range(len(classes)):
            model.classifier[1].weight[i] += float(i + 1)
            model.classifier[1].bias[i] += float(i + 1)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model)
    return path


def test_head_rows_follow_class_names_not_positions(tmp_path):
    """The regression that matters: a class inserted in the middle."""
    from ml.model import build_backbone, warm_start_head

    old = ["A", "B", "C", "D"]
    new = ["A", "B", "NEW", "C", "D"]  # inserted at index 2
    ckpt = _tiny_checkpoint(tmp_path, old)

    model = build_backbone("mobilenet_v2", len(new))
    transferred, added = warm_start_head(model, ckpt, old, new)

    assert transferred == 4
    assert added == ["NEW"]
    w = model.state_dict()["classifier.1.weight"]
    for name in old:
        expected = float(old.index(name) + 1)
        got = float(w[new.index(name)][0])
        assert got == pytest.approx(expected), (
            f"{name} got weights of {expected!r}->{got!r}; rows were matched "
            "by position, not name"
        )


def test_new_class_is_not_given_another_classes_weights(tmp_path):
    from ml.model import build_backbone, warm_start_head

    old = ["A", "B"]
    new = ["A", "NEW", "B"]
    ckpt = _tiny_checkpoint(tmp_path, old)
    model = build_backbone("mobilenet_v2", len(new))
    warm_start_head(model, ckpt, old, new)

    w = model.state_dict()["classifier.1.weight"]
    assert float(w[1][0]) not in (1.0, 2.0), "new class inherited a trained row"


def test_removed_class_does_not_shift_the_rest(tmp_path):
    from ml.model import build_backbone, warm_start_head

    old = ["A", "GONE", "B"]
    new = ["A", "B"]
    ckpt = _tiny_checkpoint(tmp_path, old)
    model = build_backbone("mobilenet_v2", len(new))
    transferred, added = warm_start_head(model, ckpt, old, new)

    assert transferred == 2 and added == []
    w = model.state_dict()["classifier.1.weight"]
    assert float(w[0][0]) == pytest.approx(1.0)   # A
    assert float(w[1][0]) == pytest.approx(3.0)   # B keeps ITS row, not GONE's


def test_mismatched_class_count_is_refused_rather_than_guessed(tmp_path):
    from ml.model import build_backbone, warm_start_head

    ckpt = _tiny_checkpoint(tmp_path, ["A", "B", "C"])
    model = build_backbone("mobilenet_v2", 2)
    with pytest.raises(ValueError, match="refusing to guess"):
        warm_start_head(model, ckpt, ["A", "B"], ["A", "B"])


def _write_export(root, rows):
    """rows: (split, filename, set-of-columns-that-are-1)."""
    cols = ["downy_mildew", "growing", "health", "raising_seeding", "sclerotinia_rot"]
    from PIL import Image

    by_split = {}
    for split, name, on in rows:
        by_split.setdefault(split, []).append((name, on))
    for split, items in by_split.items():
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        with (d / "_classes.csv").open("w", newline="") as fh:
            # Roboflow pads headers with a leading space; keep that quirk.
            w = csv.writer(fh)
            w.writerow(["filename"] + [" " + c for c in cols])
            for name, on in items:
                Image.new("RGB", (8, 8)).save(d / name)
                w.writerow([name] + ["1" if c in on else "0" for c in cols])


def test_growth_stages_are_not_treated_as_diagnoses(tmp_path):
    from ml.sources import load_lettuce_roboflow

    _write_export(tmp_path, [
        ("train", "a.jpg", {"growing", "health"}),
        ("train", "b.jpg", {"health", "raising_seeding"}),
        ("train", "c.jpg", {"downy_mildew"}),
        ("valid", "d.jpg", {"sclerotinia_rot"}),
    ])
    got = {s.path.name: s.label for s in load_lettuce_roboflow(tmp_path)}
    assert got == {
        "a.jpg": "Lettuce___healthy",
        "b.jpg": "Lettuce___healthy",
        "c.jpg": "Lettuce___Downy_mildew",
        "d.jpg": "Lettuce___Sclerotinia_rot",
    }


def test_disease_beats_health_when_both_are_marked(tmp_path):
    """Never label a plant healthy when a disease is also flagged."""
    from ml.sources import load_lettuce_roboflow

    _write_export(tmp_path, [("train", "x.jpg", {"health", "growing", "downy_mildew"})])
    samples = load_lettuce_roboflow(tmp_path)
    assert [s.label for s in samples] == ["Lettuce___Downy_mildew"]


def test_every_roboflow_sample_is_vertical_domain(tmp_path):
    from ml.sources import load_lettuce_roboflow

    _write_export(tmp_path, [("train", "a.jpg", {"health"}), ("test", "b.jpg", {"downy_mildew"})])
    assert {s.domain for s in load_lettuce_roboflow(tmp_path)} == {"vertical"}


def test_taxonomy_still_validates_after_the_new_classes():
    from ml.taxonomy import validate_taxonomy

    validate_taxonomy()  # raises if a class is unreachable or maps somewhere bogus
