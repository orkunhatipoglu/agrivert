# ml/ — training and the shared serving contract

This package is the single source of truth for anything training and serving
must agree on. The backend imports it directly; there are no copies.

## Using the model without training it

**You almost certainly want this section.** Training needs a GPU, ~10GB of
datasets and a few hours. Everyone except the person who trains it should
just download the result:

```bash
python -m ml.bundle fetch <bundle-url> --into backend/models --sha256 <hash>
```

That downloads, verifies the archive hash and every file hash inside it, and
installs it as a model version the backend resolves immediately. A truncated
download fails loudly rather than serving a model that loads and predicts
garbage.

To publish a bundle after training:

```bash
python -m ml.bundle pack artifacts/vertical --version v2-vertical-20260809
# upload dist/v2-vertical-20260809.tar.gz as a GitHub Release asset
```

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
| `--target-selective-accuracy` | 0.90 | Accuracy the recommended threshold must reach on accepted predictions. |

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

### Known coverage gaps

`taxonomy.py::UNCOVERED_TARGETS` lists crops the product wants that **no
source dataset contains**: kale, spinach, arugula, mint, cilantro, thyme,
microgreens. Photos of those will be forced into the nearest available class.
Basil, cabbage and celery have disease classes but **no healthy class**, so
the model can never call them well. Both facts are printed at the top of
every training run and carried in `metadata.json`.
