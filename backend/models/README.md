# Model registry

One directory per model version. Each must contain exactly these three files:

```
models/
  v3-vertical-20260809/
    best.pt          # checkpoint, loaded by predict.py
    metadata.json    # classes, calibration, preprocessing, metrics
    labels.json      # class index -> crop/condition/healthy
  v4-.../
```

A directory missing any of the three is skipped by the registry and logged as
a warning, rather than half-loaded. Bundles installed by
`scripts/model-download.sh` also drop a `MANIFEST.json` here; the registry
ignores it, and it is worth keeping — it records the hash of every file, so
you can tell later whether a version on disk is still the one that was
published.

## Installing the published model

```bash
../../scripts/model-download.sh --activate
```

That is the normal path: it verifies hashes, installs the version, and points
`backend/.env` at it. See [`ml/README.md`](../../ml/README.md) for the current
release and for installing something other than the pinned one.

## Adding a model you trained yourself

```bash
python ../scripts/register_model.py ../../artifacts/vertical --version v4-vertical-20260901
python ../scripts/seed_diseases.py --version v4-vertical-20260901   # if classes changed
# then, as an admin:
curl -X POST -H "Authorization: Bearer $TOKEN" \
  localhost:8000/api/v1/admin/models/v4-vertical-20260901/activate
# restart the API and the Celery workers so they load the new version
```

No backend code changes are involved. `metadata.json` carries the
normalization constants, crop size, calibration temperature and confidence
threshold, so a retrain that changes any of those updates serving
automatically (project_context.md §2.7).

Publish it for everyone else with `scripts/model-upload.sh` — see
`ml/README.md`.

## Which version gets served

Load order: the Firestore active record → `DEFAULT_MODEL_VERSION` in
`backend/.env` → the sole version on disk. If several versions exist and none
is active, startup **fails loudly** rather than guessing, because serving an
unknown model silently is worse than not serving. That is easy to trip over
locally after installing a second bundle; `--activate` on the download script
sets `DEFAULT_MODEL_VERSION` for you.

## Why the weights are gitignored

`best.pt` is ~9 MB and the whole bundle ~8.6 MB compressed. Committing
checkpoints to git bloats history permanently, and every retrain adds another
copy that can never be removed. They ship as release bundles instead —
`ml/bundle.py` and the two scripts in `scripts/`.
