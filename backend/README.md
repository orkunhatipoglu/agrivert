# Agrivert Backend

FastAPI + Celery + Redis + Firebase. Implements the routes planned in the
`agrivert-ml` project's `ROUTES.md`, against the model trained by that same
project (see its `project_context.md` for the training pipeline this backend
consumes). Serving code is imported from the repo's `ml/` package, not copied here.

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
python -m ml.bundle fetch \
  https://github.com/orkunhatipoglu/agrivert/releases/download/v3-vertical-20260809/v3-vertical-20260809.tar.gz \
  --into models \
  --sha256 17d5d2b48d0a9cedfc77bca7c4090d3f703ed385d33a561a929dd2d1399e52dc

# Only if you trained it yourself: register a local artifacts dir instead.
python scripts/register_model.py ../artifacts/vertical-v3 --version v3-vertical-20260809 --activate

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
- **Some classes have no field training data** (`studio_only_classes`).
  Diagnoses carry `fieldValidated: false` for these; the frontend hedges
  visibly on them.
- **Field accuracy is 62.2% on the current `v3-vertical` model** (31 classes,
  deduplicated test set), up from 54.4% in v2. Studio is 99.0% and vertical
  98.8%, but studio is the unrealistic domain and vertical is now dominated by
  one greenhouse capture session — field remains the honest proxy for a photo
  taken somewhere the model has never been. The `/feedback` loop is still the
  long-term fix (`project_context.md` §3 step 6).
- **The 0.910 gate meets the 90% bar — but read the per-domain rows, not the
  pooled one.** Verified on data the threshold was never fitted on:

  | domain | coverage | n | selective accuracy |
  |---|---|---|---|
  | vertical | 96.0% | 2,177 | 99.8% (99.5–99.9) |
  | **field** | 54.4% | 243 | **91.4%** (87.2–94.3) |
  | ~~pooled~~ | 89.1% | 2,420 | ~~98.9%~~ — **do not quote** |

  The pool is 2,177 vertical images against 243 field, so its mean describes
  the easy domain. `metadata.json` carries `verified_per_domain`; the frontend
  should surface the *field* figure, because that is what a photo taken
  somewhere new resembles. Roughly 46% of field photos still come back
  `uncertain`, which is intended.

  This is the first version to meet its own bar. v2 could not: its best
  defensible lower bound was 80.7%, and an earlier version *claimed* 90.8%
  only because the threshold was fitted on the test split and that split's
  accuracy was then quoted back — circular, since the fit maximises the
  number being reported. It now fits on val, verifies on test, and judges the
  worst domain rather than the average. See `ml/README.md`.

  ```bash
  python -m ml.recalibrate backend/models/<version> --target 0.90 --write
  python -m ml.bundle pack backend/models/<version> --version <version>   # hashes change
  ```

  Exit status is 2 whenever the target is missed.

- **`confidence` is this photo's score, not a success rate.** Showing it as
  "the model is 94% sure" overstates things — a ≤10% re-crop still changes the
  predicted class on ~31% of field photos. `expected_accuracy` is the number
  describing how often an accepted prediction is actually right.
  `DiseaseClassifier(dir, tta=True)` averages 3 zoom levels for steadier
  confidence at 3x cost and no accuracy gain; it is off by default.

- **Vertical's 98.8% is dominated by one capture session.** 9,979 of the
  11,347 vertical images come from a single Roboflow greenhouse shoot. It does
  generalise — a different NFT system scores 94.4% — but on only 36 held-out
  images (CI 81.9–98.5). Treat vertical accuracy as measured in *one*
  greenhouse until a second one is collected.
- **Four requested crops have no training data at all**: kale, spinach,
  arugula, mint, cilantro, thyme, microgreens. Photos of those are forced
  into the nearest available class. See `ml/taxonomy.py::UNCOVERED_TARGETS`.
