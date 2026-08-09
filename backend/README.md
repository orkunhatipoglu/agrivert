# Agrivert Backend

FastAPI + Celery + Redis + Firebase. Implements the routes planned in
[`../ROUTES.md`](../ROUTES.md), against the model built by [`../ml/`](../ml/README.md).
Serving code is imported from `ml/`, not copied here.

> Comments throughout this codebase cite `project_context.md` by section
> number. That document is the original planning conversation and is **not in
> this repo**; `ROUTES.md` reconstructs the parts the routes depend on, and
> `ml/README.md` covers the training pipeline. Treat the citations as
> provenance, not as a file you can open.

## What is and isn't implemented

| Route group | Status |
|---|---|
| `/diagnoses/*` | **Implemented** — upload, validation, queue, inference, poll, SSE stream, image fetch, delete, feedback |
| `/admin/models`, `/admin/models/{v}/activate` | **Implemented** — model registry + activation |
| `/auth/register`, `/auth/me` | **Implemented** |
| `/health` | **Implemented** — reports model/Firestore/broker readiness |
| `/diseases`, `/diseases/{id}` | Reads implemented; **content empty by design** (see below) |
| `/auth/login`, `/refresh`, `/logout` | **501** — Firebase client SDK owns these; see `app/routers/auth.py` |
| `/admin/stats` | **501** — needs a rollup job; Firestore has no `GROUP BY` |

Every stub returns 501 with a message explaining what it needs, so nothing
fails silently or looks finished when it isn't.

## Architecture

```
POST /diagnoses  ──> validate (sync)  ──> Firebase Storage
                          │                     │
                     Firestore              object name
                     status=queued              │
                          │                     ▼
                          └──────────────> Redis queue
                                                │
                                          Celery worker
                                                │
                                    predict.py (imports data.py)
                                                │
                                    Firestore status=completed|uncertain
                                                │
GET /diagnoses/{id}  or  /stream  <─────────────┘
```

The image travels by *reference* (storage object name), not through Redis —
otherwise every diagnosis would push megabytes of JPEG through the broker.

### Why the backend imports `predict.py` instead of reimplementing inference

`predict.py` builds its eval transform from `data.build_eval_transform`, so
importing it guarantees serving preprocessing is byte-identical to training.
Reimplementing resize/crop/normalize here would reintroduce exactly the
train/serve drift `project_context.md` §2.7 warns about.

They are the **same modules**, imported from `ml/` at the repo root —
not copies. `ML_REPO_ROOT` (`app/config.py`) points at the repo root and
defaults correctly; the Dockerfile copies `ml/` into the image.

This replaced vendored copies under `backend/`. Those drifted from the
originals and produced an `ImportError` on a teammate's first run for three
names that had never existed anywhere. `ml/contract.py` now owns every
constant both sides need, and `tests/test_contract.py` fails if they split
apart again.

### Model modularity

Weights are **not** referenced by any code path. `backend/models/<version>/`
holds each version; Firestore records which is active; `resolve_active_version()`
resolves it at load time. Swapping a retrained model is a file drop plus an
activate call — see `models/README.md`.

Load order: Firestore active record → `DEFAULT_MODEL_VERSION` → the sole
version on disk. If two versions exist and none is active, startup **fails
loudly** rather than guessing, because serving an unknown model silently is
worse than not serving.

## Setup

```bash
cp .env.example .env          # fill in Firebase project + bucket
mkdir -p secrets              # put serviceAccount.json here (gitignored)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Install the published model bundle — no training, no GPU, no datasets.
# The version, URL and hash are pinned in scripts/model-release.env.
../scripts/model-download.sh --activate

# Only if you trained it yourself: register a local artifacts dir instead.
python scripts/register_model.py ../artifacts/vertical --version v3-vertical-20260809 --activate

python scripts/seed_diseases.py
```

Run locally (three terminals, or use Docker):

```bash
redis-server
celery -A app.worker.celery_app:celery_app worker --loglevel=info --concurrency=1
uvicorn app.main:app --reload
```

Or, from this directory:

```bash
docker compose up --build
```

Interactive docs at http://localhost:8000/docs.

## Tests

```bash
python -m pytest tests/ -q
```

Covers upload validation (including the truncated-JPEG case that
`Image.verify()` misses — `project_context.md` §2.9 bug #5) and registry
resolution. Firebase and torch are not exercised; those need real
credentials and a real GPU.

## Firestore setup

Collections: `users`, `diagnoses`, `diagnosis_feedback`, `diseases`,
`model_versions`.

`GET /diagnoses` filters need composite indexes. Firestore will emit a
console link with the exact index on first failing query; the common one is
`owner_uid ASC, created_at DESC`.

Grant admin rights with a custom claim:

```python
from firebase_admin import auth
auth.set_custom_user_claims(uid, {"admin": True})
```
