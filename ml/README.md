# ml/ — training and the shared serving contract

This package is the single source of truth for anything training and serving
must agree on. The backend imports it directly; there are no copies.

## Using the model without training it

**You almost certainly want this section.** Training needs a GPU, ~10GB of
datasets and a few hours. Everyone except the person who trains it should
just download the result:

```bash
python -m ml.bundle fetch \
  https://github.com/orkunhatipoglu/agrivert/releases/download/v3-vertical-20260809/v3-vertical-20260809.tar.gz \
  --into backend/models \
  --sha256 17d5d2b48d0a9cedfc77bca7c4090d3f703ed385d33a561a929dd2d1399e52dc
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
| Tag | `v3-vertical-20260809` |
| Classes | 31 (adds `Lettuce___Sclerotinia_rot`) |
| Accuracy | studio 99.0% / **field 62.2%** / vertical 98.8% |
| Confidence threshold | 0.910 — field 54.4% answered at 91.4%; vertical 96.0% at 99.8% |
| Meets its 90% bar | **Yes**, under the strict worst-domain rule |

Releases: <https://github.com/orkunhatipoglu/agrivert/releases>

### Publishing a bundle after training

```bash
python -m ml.bundle pack artifacts/vertical-v3 --version v3-vertical-20260809
```

That writes three files to `dist/` — `.tar.gz`, `.tar.gz.sha256` and
`.manifest.json`. **Attach all three** to a GitHub Release tagged with the same
version string, so the download URL above resolves. Then update the hash in
this file and in `backend/README.md`.

```bash
gh release create v3-vertical-20260809 dist/v3-vertical-20260809.* \
  --title "Vertical-ag model v3" \
  --notes "31 classes. studio 99.0% / field 62.2% / vertical 98.8%. Gate 0.910: field 91.4% on 54.4% coverage."
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
| `--vertical-boost` | 12.0 | Oversampling of the vertical domain. Sized for when vertical was 209 images; it now has 11,347, so **2.0 (~32% of each epoch) is the tested v3 setting**. At 12 it would swamp field entirely. |
| `--selection-domains` | `vertical,field` | Domains whose mean macro-F1 picks the checkpoint and fits calibration. |
| `--vertical-holdout` | 0.2 | Vertical uses 60/20/20; studio and field use 80/10/10. |
| `--no-dedup` | off | Split per image instead of per subject. Diagnostic only — it inflates every vertical number. |
| `--dedup-threshold` | 5 | dHash Hamming distance (of 64) counted as the same subject. |
| `--target-selective-accuracy` | 0.90 | Accuracy the recommended threshold must reach on accepted predictions — judged on the 95% CI **lower bound**, fitted on val, verified on test. Judged per domain, not pooled — see below. |
| `--min-accepted` | 40 | Ignore gates accepting fewer val images than this. 100% accuracy on 6 photos is not a measurement. |

### Fine-tuning onto an expanded taxonomy

```bash
python -m ml.train_vertical --download --init-from artifacts/vertical \
  --out-dir artifacts/vertical-v3 \
  --sources plantvillage,plantdoc,plantwild,plantseg,lettuce_hydroponic,lettuce_kaggle,lettuce_roboflow \
  --root lettuce_roboflow=~/.cache/agrivert/lettuce_roboflow/extracted \
  --vertical-boost 2.0
```

`--init-from` carries the backbone over whole and matches head rows **by
class name**. Matching by position is the trap: insert a class anywhere but
the end and every later class silently inherits its neighbour's weights.
Nothing errors, loss still falls, and the model just starts from a worse
place than it should. Adding `Lettuce___Sclerotinia_rot` in the middle of the
lettuce block moved 25 of 30 classes by one index — all 25 kept their own
weights, and `backend/tests/test_finetune.py` fails if that ever regresses.

Classes present in the checkpoint but gone from the taxonomy are dropped with
a warning; classes new to the taxonomy start fresh.

### Sources, and which ones are worth their download

| Source | Domain | Images | Verdict |
|---|---|---|---|
| `plantvillage` | studio | 22,200 | Core studio set |
| `plantdoc` / `plantwild` / `plantseg` | field | 4,521 | The honest domain |
| `lettuce_hydroponic` | vertical | 209 | Only true vertical-farm source |
| `lettuce_roboflow` | vertical | 9,979 | **The one that mattered** |
| `lettuce_kaggle` | vertical | 1,159 of 2,337 | Partly usable |
| ~~`plant-disease-expert`~~ | — | 200,506 | **Rejected — see below** |

`lettuce_roboflow` (Roboflow `phs/lettuce_disease`, MIT) took the vertical
domain from 209 images to 11,347 and added `Lettuce___Sclerotinia_rot`. It
has no auto-download: a Roboflow *version* has to be generated before an
export exists, so pass `--root lettuce_roboflow=<unzipped export>`. Generate
it with 640px "Fit within" and **no augmentation** — augmentation belongs in
training, and a stretched resize would feed distorted aspect ratios into a
pipeline that centre-crops. The export's `growing` and `raising_seeding`
columns are growth stages that co-occur with `health`, not conditions; see
`LETTUCE_ROBOFLOW_MAP`.

Read the counts before trusting a dataset's description:

* **`ashishjstar/lettuce-diseases`** advertises 8 lettuce disease classes in
  1.25GB. 1,123 images are Healthy and 1,106 are *Shepherd's purse*, a weed,
  shipped as 119x119 thumbnails. The five disease folders hold 6-30 images
  each. Only the three folders mapping onto classes we already have are
  loaded; a class built from 6 images cannot be learned but is more than
  enough to produce confident nonsense.
* **`sadmansakibmahi/plant-disease-expert`** is 10.6GB and 200,506 images,
  and it is PlantVillage re-augmented. Every class is an exact multiple of
  its PlantVillage counterpart — 6048/630 and 11328/1180 are both 9.6x,
  5727/1909 and 3000/1000 are both 3.0x. Independent collection does not
  produce exact integer ratios against another dataset. Adding it would pour
  200k more studio images into the domain that is already at 96.8% and
  already over-represented, while doing nothing for field or vertical. Its
  genuinely novel content is thin: N deficiency (33), K deficiency (54),
  waterlogging (21), cabbage looper (234), plus tea/rice/garlic crops outside
  this taxonomy. The deficiency folders are *generic plants*, not lettuce, so
  mapping them onto the lettuce deficiency classes would corrupt two of the
  four classes the vertical product actually depends on.

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

### How the confidence gate is chosen

Three rules, each added after the previous version of this shipped a number
that was flattering rather than true. All three are enforced by
`backend/tests/test_calibration.py`.

**1. Fit on val, verify on test.** The threshold used to be fitted on the
*test* split, and that split's selective accuracy then reported as the
expected quality. The report could not fail — the number quoted was exactly
what the fit maximised. The v2 gate measured properly:

| 0.91 gate (v2) | coverage | selective accuracy |
|---|---|---|
| test split (where it was fitted) | 22.3% | **90.8%** ← the claim |
| val split (never seen by the fit) | 17.7% | **86.4%** ← reality |

**2. Judge by the CI lower bound, not the point estimate.** With ~90 accepted
images a measured 91% reaches below 83%, so selecting on the point estimate
picks whichever threshold got lucky. `--min-accepted` (default 40) discards
gates too small to measure at all — 100% on 6 photos is not a measurement.

**3. Judge the worst domain, not the average** (`--gate-rule worst-domain`,
the default). This one is new in v3 and it matters most. Pooling vertical and
field was sound while vertical was 41 images against field's 447. Importing
9,979 Roboflow images inverted that: the pool became 84% vertical, and
vertical is easy. At threshold 0.30 the pooled figure read **95.3%** while a
field photo accepted at that same gate was right **74.0%** of the time. The
average was hiding the domain a grower's photo actually resembles. A gate now
qualifies only if *every* selection domain clears the target on its own.

The v3 gate is **0.910**, and this is the first version to **meet** the 90%
bar (v2 could not reach it at all — its best defensible lower bound was
80.7%). Verified on test, never fitted on:

| domain | coverage | n | selective accuracy |
|---|---|---|---|
| vertical | 96.0% | 2,177 | 99.8% (99.5–99.9) |
| **field** | 54.4% | 243 | **91.4%** (87.2–94.3) |
| ~~pooled~~ | 89.1% | 2,420 | ~~98.9%~~ — do not quote |

`metadata.json` carries `verified_per_domain` for exactly this reason. Quote
the per-domain rows; the pooled mean is 2,177 images against 243 and tells
you about the larger one.

```bash
python -m ml.recalibrate artifacts/vertical-v3 --target 0.90 --write
python -m ml.recalibrate artifacts/vertical-v3 --gate-rule pooled   # the old, flattering view
```

Recalibration reads `split_config` from the model's own metadata, so it
rebuilds the exact split the run trained against. Passing the source list by
hand is how a threshold ends up fitted on data the model trained on — it
produces a plausible number and no error. Exit status is 2 when the target is
missed, so CI catches a regression that would otherwise surface only as a
slightly nicer-looking number.

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

The v3 data helped this partially and honestly not completely: class flips
under a ≤10% crop fell from **40.0% to 30.8%**, but the confidence *swings*
grew slightly (p95 0.296 → 0.391), because v3's temperature is 0.715 rather
than 0.98 and a sharper distribution moves further. The gate is what actually
protects a grower here: at 0.910 only 54.4% of field photos are answered at
all, and those are 91.4% correct.

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
