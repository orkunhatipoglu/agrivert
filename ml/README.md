# ml/ — training and the shared serving contract

This package is the single source of truth for anything training and serving
must agree on. The backend imports it directly; there are no copies.

## Using the model without training it

**You almost certainly want this section.** Training needs a GPU, ~10GB of
datasets and a few hours. Everyone except the person who trains it should
just download the result:

```bash
python -m ml.bundle fetch \
  https://github.com/orkunhatipoglu/agrivert/releases/download/v2-vertical-20260809/v2-vertical-20260809.tar.gz \
  --into backend/models \
  --sha256 3faba3371886d2d7124337613f0d3aec3e7b7270f71be256e61b1f8fcde78def
```

That is the whole install. No GPU, no Kaggle account, no dataset downloads.
It downloads, verifies the archive hash and every file hash inside it, and
installs it as a model version the backend resolves immediately. A truncated
download fails loudly rather than serving a model that loads and predicts
garbage — which is why the `--sha256` is worth pasting rather than omitting.

The hash above belongs to **that specific tag**. Both change together on every
release, so copy the pair from the release page you are installing rather than
mixing a new URL with an old hash — `bundle fetch` will refuse the mismatch.

### Current release

| | |
|---|---|
| Tag | `v2-vertical-20260809` |
| Size | 8.6 MB |
| Classes | 30 |
| Accuracy | studio 96.8% / field 54.4% / vertical 95.1% |
| Confidence threshold | 0.955 — answers 14.8% of photos at 95.8% (95% CI 88.5–98.6) |
| Meets its 90% bar | **No.** See "The threshold does not meet the declared bar" below |
| sha256 | `3faba3371886d2d7124337613f0d3aec3e7b7270f71be256e61b1f8fcde78def` |

Releases: <https://github.com/orkunhatipoglu/agrivert/releases>

### Publishing a bundle after training

```bash
python -m ml.bundle pack artifacts/vertical --version v2-vertical-20260809
```

That writes three files to `dist/` — `.tar.gz`, `.tar.gz.sha256` and
`.manifest.json`. **Attach all three** to a GitHub Release tagged with the same
version string, so the download URL above resolves. Then update the hash in
this file and in `backend/README.md`.

```bash
gh release create v2-vertical-20260809 dist/v2-vertical-20260809.* \
  --title "Vertical-ag model v2" \
  --notes "30 classes. studio 96.8% / field 54.4% / vertical 95.1% (n=41). Threshold 0.955."
```

Without `gh` installed, the same thing via
<https://github.com/orkunhatipoglu/agrivert/releases/new>.

## Files

| File | What it owns |
|---|---|
| `contract.py` | Preprocessing constants, label-map construction, checkpoint format, the required-artifact list. **Nothing here may be re-derived elsewhere.** |
| `model.py` | `build_backbone`, `save_checkpoint`, `load_checkpoint_into` |
| `taxonomy.py` | The vertical-ag class vocabulary and per-source label maps |
| `sources.py` | One adapter per dataset; quarantines each source's quirks |
| `dedup.py` | Perceptual hashing + grouping, so a split measures generalisation |
| `data.py` | Dataset scanning, augmentation, image loading |
| `train_vertical.py` | The vertical-agriculture training run |
| `train_mobilenetv2.py` | The older 38-class PlantVillage+PlantDoc run, kept for reference |
| `predict.py` | `DiseaseClassifier` — what the backend serves with |
| `recalibrate.py` | Re-fit the confidence threshold on a trained model, no retraining |
| `bundle.py` | Pack/fetch model bundles |

## Why `contract.py` exists

A teammate's first run died on:

```
ImportError: cannot import name 'DEFAULT_CENTER_CROP' from 'data'
```

`bootstrap_dev_model.py` imported `build_label_map`, `DEFAULT_RESIZE` and
`DEFAULT_CENTER_CROP` — none of which existed. The label-map logic had been
pasted inline into the training script, the resize ratio was a bare `1.14`
inside a transform builder, and `_build_backbone` was imported from a module
that never defined it. On top of that, the bootstrap saved a bare
`state_dict()` while the loader expected `checkpoint["model"]`.

Five breakages, one cause: **nothing declared the contract**, so each file
invented its own version of it. `contract.py` and `model.py` are now the only
definitions, and `backend/tests/test_contract.py` fails if they drift apart
again.

## Training

```bash
python -m ml.train_vertical --download --out-dir artifacts/vertical --vertical-boost 4.0
python -m ml.train_vertical --download --dry-run     # inspect the blend first
```

`--dry-run` scans the sources, prints the class/domain composition, the
near-duplicate counts and the effective per-epoch domain mix, then stops.
**Always run it first.** It is where you find out a source contributed
nothing — which is exactly how PlantWild sat at 0 images while the run still
reported success.

### Flags worth knowing

| Flag | Default | Why you would touch it |
|---|---|---|
| `--vertical-boost` | 12.0 | Oversampling of the vertical domain. At 12 it takes ~65% of every epoch from 126 images (~111 repeats each) to serve 4 of 30 classes. **4.0 (~38%) is the tested setting.** |
| `--selection-domains` | `vertical,field` | Domains whose mean macro-F1 picks the checkpoint and fits calibration. |
| `--vertical-holdout` | 0.2 | Vertical uses 60/20/20; studio and field use 80/10/10. |
| `--no-dedup` | off | Split per image instead of per subject. Diagnostic only — it inflates every vertical number. |
| `--dedup-threshold` | 5 | dHash Hamming distance (of 64) counted as the same subject. |
| `--target-selective-accuracy` | 0.90 | Accuracy the recommended threshold must reach on accepted predictions — judged on the 95% CI **lower bound**, fitted on val, verified on test. This model does not reach it; see below. |
| `--min-accepted` | 40 | Ignore gates accepting fewer val images than this. 100% accuracy on 6 photos is not a measurement. |

### Domains

`studio` (PlantVillage) / `field` (PlantDoc, PlantWild, PlantSeg) /
`vertical` (hydroponic + greenhouse lettuce).

Selection and calibration use **vertical + field pooled**, not vertical
alone. Vertical is the deployment domain, but it is ~40 held-out images
covering 4 of the 30 classes — far too small and too narrow to steer a run
by itself. Selecting on it alone stopped a run at epoch 3 on a lucky peak
while the field score was still climbing for another 5 epochs, and it fitted
a confidence threshold of **0.3**, at which pooled selective accuracy is
only 63%. Pooling keeps the target domain in charge without letting 40
images end the run or set the gate.

### Why the split is grouped, not random

The hydroponic lettuce source is burst photography — `IMG20251222130237.jpg`
and `IMG20251222130242.jpg` are the same plant five seconds apart. PlantWild
and PlantSeg overlap heavily with each other. Measured with `ml.dedup`:

| Domain | Images | Distinct subjects | Near-duplicates |
|---|---|---|---|
| studio | 22,200 | 21,963 | 1% |
| field | 4,521 | 2,573 | **43%** |
| vertical | 209 | 185 | 11% |

Split those at random and val is largely re-photographs of train. The run
then reads ~0.96 vertical F1 by epoch 1 having learned nothing transferable,
and — the part that actually hurts — the confidence threshold is fitted
against that fiction, so serving accepts predictions it should refuse.
`ml.dedup` groups near-duplicates and the split allocates whole groups.

Report the leakage in any dataset directly:

```bash
python -m ml.dedup /path/to/dataset --threshold 5
```

### The threshold does not meet the declared bar

`--target-selective-accuracy` is 0.90. **No threshold reaches it.** The best
any gate can defend on validation data is a 80.7% lower bound. `metadata.json`
records `meets_target: false`, `predict.py` returns it on every diagnosis, and
nothing in this project may advertise 90%.

This was hidden by a calibration bug worth understanding, because the shape of
it recurs. The threshold was **fitted on the test split**, and that same
split's selective accuracy was then reported as the expected quality. The
report could not fail: the number being quoted was exactly the quantity the
fit maximised. Measured properly, the shipped 0.91 gate looked like this:

| | coverage | selective accuracy |
|---|---|---|
| test split (where it was fitted) | 22.3% | **90.8%** ← the claim |
| val split (never seen by the fit) | 17.7% | **86.4%** ← reality |

Two rules now prevent it, both enforced by `backend/tests/test_calibration.py`:

* **Fit on val, verify on test.** `metadata.json` carries `threshold_fitted_on`
  and `threshold_verified_on`, and `calibration.verified` holds the only
  selective-accuracy figure that is a measurement rather than a fitted
  quantity. Quote that one.
* **Judge a gate by the lower bound of its 95% CI, not the point estimate.**
  With ~90 accepted images a measured 91% reaches below 83%; selecting on the
  point estimate picks whichever threshold got lucky, and luck does not
  survive new photos. `--min-accepted` (default 40) additionally discards
  gates too small to measure at all — 100% on 6 photos is not a measurement.

The shipped gate uses `--policy conservative`: the most selective threshold
still measurable, chosen because a confident wrong answer to a grower costs
more than an honest "I can't tell". It answers **14.8%** of photos at **95.8%**
(95% CI 88.5–98.6, n=72 verified on test). Most photos return `uncertain`.
That is the intended behaviour.

```bash
python -m ml.recalibrate artifacts/vertical --target 0.90 --policy conservative --write
python -m ml.recalibrate artifacts/vertical --target 0.80                        # what 80% would buy
```

Exit status is 2 when the target is missed, so CI can catch a regression that
would otherwise only show up as a slightly nicer-looking number.

### Why confidence moves when you re-crop a photo

Re-cropping a photo by a few percent can swing its confidence by 0.3 and
change the predicted class outright. On held-out field photos, **31% changed
class under a ≤10% crop**, with a mean confidence swing of 0.15.

Two causes, and the first is not a bug so much as a consequence:

1. **Eval keeps only the centre square.** `build_eval_transform` resizes the
   short side to 255 and centre-crops 224, so a 4:3 photo loses ~42% of its
   area and a 16:9 photo ~57%. Re-cropping slides that window over different
   pixels.
2. **The model is ~55% accurate on field photos.** Near the decision boundary
   a small input change flips the argmax. This dominates, and no preprocessing
   change fixes it — only a better model does.

`build_tta_transforms` averages three zoom levels and cuts flips to ~20% and
the mean swing to ~0.11. It is **off by default** and worth being precise
about what it buys: 3× inference cost, materially steadier confidence, and
**no accuracy gain** — on val it measured slightly *worse* (58.4% vs 60.8%).

```python
DiseaseClassifier(model_dir, tta=True)      # steadier, 3x cost
```

Feeding the whole frame instead (`squash`/`pad-to-square`) is worse still —
45.0% and 50.1% field accuracy against 54.4% — because the model was trained
on `RandomResizedCrop` views and a full frame is out of distribution for it.

### Known coverage gaps

`taxonomy.py::UNCOVERED_TARGETS` lists crops the product wants that **no
source dataset contains**: kale, spinach, arugula, mint, cilantro, thyme,
microgreens. Photos of those will be forced into the nearest available class.
Basil, cabbage and celery have disease classes but **no healthy class**, so
the model can never call them well. Both facts are printed at the top of
every training run and carried in `metadata.json`.
