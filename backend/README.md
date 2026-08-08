# Agrivert Backend

FastAPI + Celery + Redis + Firebase. Implements the routes planned in the
`agrivert-ml` project's `ROUTES.md`, against the model trained by that same
project (see its `project_context.md` for the training pipeline this backend
consumes). `predict.py` and `data.py` here are vendored, byte-identical
copies from that repo — this `backend/` directory is self-contained and
buildable on its own.

## What is and isn't implemented

| Route group | Status |
|---|---|
| `/diagnoses/*` | **Implemented** — upload, validation, queue, inference, poll, SSE stream, image fetch, delete, feedback |
| `/farms/*` | **Implemented** — farm and plot CRUD, cascade delete, ownership-scoped |
| `/admin/models`, `/admin/models/{v}/activate` | **Implemented** — model registry + activation |
| `/auth/register`, `/auth/me` | **Implemented** |
| `/health` | **Implemented** — reports model/Firestore/broker readiness |
| `/diseases`, `/diseases/{id}` | Reads implemented; **content empty by design** (see below) |
| `/auth/login`, `/refresh`, `/logout` | **501** — Firebase client SDK owns these; see `app/routers/auth.py` |
| `/admin/stats` | **501** — needs a rollup job; Firestore has no `GROUP BY` |

Every stub returns 501 with a message explaining what it needs, so nothing
fails silently or looks finished when it isn't.

`/notifications` was removed rather than left stubbed: regional outbreak
alerts need farm locations and enough diagnosis volume per region to make
"trending" mean anything, and neither exists yet.

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

`predict.py` and `data.py` are **vendored into this directory** rather than
imported from a sibling repo path — the training project (`agrivert-ml`) and
this backend are pushed to different places (the training repo is not on
GitHub; this backend is), so a cross-repo relative path would only work on
one machine. `ML_REPO_ROOT` (`app/config.py`) defaults to `backend/` itself;
override it only if you relocate these files.

**Keeping them in sync is manual.** When the training pipeline changes
`predict.py` or `data.py` — especially `build_eval_transform`, since serving
must match training exactly — copy the updated files into `backend/` again.
Nothing currently automates or checks this; a staleness check (e.g. comparing
file hashes at startup) would be a reasonable thing to add if the two drift
in practice.

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

# Path to the artifacts/ dir is wherever your training run's output lives —
# adjust if this isn't checked out next to the agrivert-ml project.
python scripts/register_model.py /path/to/agrivert-ml/artifacts --version v1-blended-20260808 --activate
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

Collections: `users`, `farms`, `plots`, `diagnoses`, `diagnosis_feedback`,
`diseases`, `model_versions`.

`GET /diagnoses` filters need composite indexes. Firestore will emit a
console link with the exact index on first failing query; the common one is
`owner_uid ASC, created_at DESC`.

Grant admin rights with a custom claim:

```python
from firebase_admin import auth
auth.set_custom_user_claims(uid, {"admin": True})
```

## Known gaps / decisions to revisit

- **`predict.py`/`data.py` are vendored copies, not a shared package.** They
  must be re-copied from `agrivert-ml` after any change there, especially to
  `build_eval_transform`. `scripts/check_vendored.py <agrivert-ml-path>`
  compares them by hash and can `--update` in place; run it after pulling
  training changes.
- **The disease KB ships empty.** `seed_diseases.py` creates a document per
  class with blank `description`/`symptoms`/`treatment` and
  `content_reviewed: false`. This is deliberate: that text is what a farmer
  reads before treating a real crop, and generating it without a cited
  agronomic source would produce fluent, unreviewed advice indistinguishable
  from reviewed advice. Fill it in, then flip `content_reviewed` — the
  `recommendation` field on a diagnosis stays null until you do.
- **`GET /admin/stats` is still a 501.** Firestore has no `GROUP BY`, so it
  needs either counters incremented on write or a scheduled rollup job.
- **`POST /auth/login|refresh|logout` are 501 by design** — the Firebase
  client SDK owns the credential lifecycle. Only revisit if you move off
  Firebase Auth.
- **10 of 38 classes have no field training data** (`studio_only_classes`).
  Diagnoses carry `fieldValidated: false` for these; the frontend hedges
  visibly on them.
- **Field accuracy is 65.3%.** The `/feedback` loop is the long-term fix
  (`project_context.md` §3 step 6). Until then, the 0.95 confidence
  threshold means a large share of real photos will come back `uncertain` —
  that is the intended behavior, not a bug.
