# Model registry

One directory per model version. Each must contain exactly these three files:

```
models/
  v1-blended-20260808/
    best.pt          # checkpoint, loaded by predict.py
    metadata.json    # classes, calibration, preprocessing, metrics
    labels.json      # class index -> crop/condition/healthy
  v2-.../
```

A directory missing any of the three is skipped by the registry and logged as
a warning, rather than half-loaded.

## Adding a retrained model

```bash
python scripts/register_model.py ../artifacts --version v2-blended-20260901
python scripts/seed_diseases.py --version v2-blended-20260901   # if classes changed
# then, as an admin:
curl -X POST -H "Authorization: Bearer $TOKEN" \
  localhost:8000/api/v1/admin/models/v2-blended-20260901/activate
# restart the Celery workers so they load the new version
```

No backend code changes are involved. `metadata.json` carries the
normalization constants, crop size, calibration temperature and confidence
threshold, so a retrain that changes any of those updates serving
automatically (project_context.md §2.7).

## Why the weights are gitignored

`best.pt` is ~26MB. Committing checkpoints to git bloats history permanently
and GitHub's Contents API rejects files that size anyway. Ship them via the
image build, a mounted volume, Git LFS, or a release asset.
