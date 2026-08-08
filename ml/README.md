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
| `data.py` | Dataset scanning, augmentation, image loading |
| `train_vertical.py` | The vertical-agriculture training run |
| `train_mobilenetv2.py` | The older 38-class PlantVillage+PlantDoc run, kept for reference |
| `predict.py` | `DiseaseClassifier` — what the backend serves with |
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
python -m ml.train_vertical --download --out-dir artifacts/vertical
python -m ml.train_vertical --dry-run --root plantwild=/data/pw   # inspect the blend first
```

`--dry-run` scans the sources, prints the class/domain composition and the
effective per-epoch domain mix, then stops. Run it before committing to a
full training run — it is where you find out a source contributed nothing.

### Domains

`studio` (PlantVillage) / `field` (PlantDoc, PlantWild, PlantSeg) /
`vertical` (hydroponic + greenhouse lettuce). Checkpoint selection,
temperature scaling and the confidence threshold all target **vertical**,
because that is the deployment domain and the domain you select on is the
domain you get.

### Known coverage gaps

`taxonomy.py::UNCOVERED_TARGETS` lists crops the product wants that **no
source dataset contains**: kale, spinach, arugula, mint, cilantro, thyme,
microgreens. Photos of those will be forced into the nearest available class.
Basil, cabbage and celery have disease classes but **no healthy class**, so
the model can never call them well. Both facts are printed at the top of
every training run and carried in `metadata.json`.
