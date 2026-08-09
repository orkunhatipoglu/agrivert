# Technical stack

What Agrivert is built out of, and why each piece is there. Versions are the
ones actually declared in `frontend/package.json`, `backend/requirements.txt`,
`ml/requirements.txt` and `backend/Dockerfile` — not aspirational ones.

## At a glance

| Layer | Technology | Version |
|---|---|---|
| Client | Next.js (App Router) + React | `16.2.6` / `19.2.4` |
| Styling | Tailwind CSS + shadcn/ui on Radix | `4.x` / `radix-ui 1.6` |
| Client state | TanStack Query | `5.101` |
| API | FastAPI + Uvicorn | `>=0.115` / `>=0.30` |
| Validation | Pydantic + pydantic-settings | `>=2.8` / `>=2.4` |
| Queue | Celery + Redis | `>=5.4` / redis `7-alpine` |
| Inference | PyTorch + torchvision | `>=2.4` / `>=0.19` |
| Preprocessing | Albumentations + OpenCV (headless) + Pillow | `>=1.4.15` / `>=4.9` / `>=10.0` |
| Model | MobileNetV2, 224px, 31 classes | `v3-vertical-20260809` |
| Database | Cloud Firestore | via `firebase-admin >=6.5` |
| Object storage | Firebase Storage | same SDK |
| Auth | Firebase Auth (client SDK issues, admin SDK verifies) | `firebase 12.17` / `firebase-admin >=6.5` |
| Container | `python:3.11-slim`, one image for API and worker | — |

Three processes plus Redis, and a fourth in development:

```
next dev (3000) ──HTTP──> uvicorn (8000) ──> Redis ──> celery worker
                              │                            │
                              └────── Firestore + Storage ──┘
```

## Runtime requirements

| | Requirement | Enforced by |
|---|---|---|
| Python | 3.11 or 3.12 | `start.sh` prefers 3.12; the image is 3.11-slim |
| Node | >= 20 | `start.sh` refuses to continue below it — Next 16 needs it |
| Redis | 7 | `docker-compose.yml`; `start.sh` will use a local or dockerised one |
| GPU | optional | `inference_device` defaults to cuda-if-available, else CPU |

`start.sh` installs missing system packages via brew/apt and builds
`backend/.venv` with `uv` when available.

## Frontend

Next.js 16 App Router with two route groups: `(auth)` for signed-out pages and
`(app)` for everything behind `AuthGuard`. React 19, TypeScript 5, Tailwind v4
via `@tailwindcss/postcss`, and shadcn/ui components (~30 of them under
`components/ui/`) built on `radix-ui` primitives. `next-themes` for the theme,
`sonner` for toasts, `lucide-react` for icons, `date-fns` for formatting.

TanStack Query owns server state. Two details are load-bearing:

- **`lib/api.ts` fetches a fresh ID token per request** rather than caching
  one. Firebase's `getIdToken()` handles refresh internally, so caching a
  token in the client would only add a way to send an expired one.
- **SSE goes through `fetch`, not `EventSource`** (`hooks/use-diagnosis-live.ts`).
  `EventSource` cannot set an `Authorization` header, and the stream endpoint
  is authenticated like every other route. It falls back to polling.

Images are fetched as blobs for the same reason: `<img src>` cannot
authenticate, so `components/diagnosis-image.tsx` fetches and object-URLs it.

## Backend API

FastAPI under an `/api/v1` prefix, five routers (`auth`, `diagnoses`,
`diseases`, `admin`, `health`). Pydantic v2 schemas in `app/schemas/`,
Firestore access isolated in `app/repositories/`, and `pydantic-settings`
reading `backend/.env` from an absolute path so the same config loads whether
the process starts in `backend/`, the repo root, or under pytest.

CORS is explicit and `"*"` is **rejected at startup** rather than warned
about: the API sends credentialed requests, and browsers refuse a wildcard
with credentials, so a permissive default would fail confusingly at runtime
instead of loudly at boot.

Upload validation happens synchronously before anything is queued —
12 MB cap, `image/jpeg|png|webp`, minimum 64px — because failing deep inside
inference gives a farmer a useless error minutes later.

## Queue

Celery over Redis, `--concurrency=1`, scaled with `--scale worker=N` rather
than more threads: each process loads its own copy of the checkpoint, so
concurrency multiplies memory rather than throughput.

**The image travels by reference.** `POST /diagnoses` writes the file to
Firebase Storage and puts only the object name on the queue; otherwise every
diagnosis would push megabytes of JPEG through the broker.

**grpc must not be imported in the Celery parent process.** Celery imports
`app.worker.tasks` in the parent and then forks its pool children. grpc is not
fork-safe: a child forked from a parent that already imported it inherits a
dead c-ares resolver, and every Firestore call then burns its full retry
deadline before failing with a DNS error. Tasks appear to hang forever while
the API — which never forks — talks to Firestore fine. Nothing in that failure
points at an import statement, so `tests/test_worker_fork_safety.py` fails the
moment a top-level Firestore import returns to `app/worker/tasks.py`.

## Inference

`app/ml/engine.py` wraps `ml.predict.DiseaseClassifier`; it does not
reimplement anything. `predict.py` builds its eval transform from
`ml.data.build_eval_transform`, so serving preprocessing is byte-identical to
training preprocessing — the same objects, not a copy that can drift. The
classifier is cached per version behind a lock and loaded lazily, so a worker
pays the checkpoint load once.

Model parameters are **not in the code**. Normalization constants, crop size,
calibration temperature and confidence threshold are read from each version's
`metadata.json` at load time, so a retrain that changes any of them updates
serving without a code change. `app/ml/registry.py` resolves which version to
serve: Firestore's active record → `DEFAULT_MODEL_VERSION` → the sole version
on disk, and it **fails loudly** if several exist and none is active.

Weights ship as release bundles, never in git — see
[`ml/README.md`](ml/README.md) and `scripts/model-download.sh`.

## Data and auth

Firestore collections: `users`, `diagnoses`, `diagnosis_feedback`,
`diseases`, `model_versions`. No ORM and no migrations;
`app/repositories/` is the only layer that touches the client. Composite
indexes are needed for the `GET /diagnoses` filters — the common one is
`owner_uid ASC, created_at DESC`, and Firestore emits a console link with the
exact index on first failing query.

Auth is split by design. The **client SDK owns the credential lifecycle**
(sign-in, refresh, sign-out), and the API only verifies: `verify_id_token(...,
check_revoked=True)` in `app/dependencies.py`. That is why
`POST /auth/login|refresh|logout` return 501 — they would be a worse
reimplementation of something the SDK already does correctly. Admin rights are
a Firebase custom claim.

## ML training

| | |
|---|---|
| Architecture | MobileNetV2, ImageNet-pretrained, 224×224 centre crop (`build_backbone` also supports `mobilenet_v3_large`) |
| Framework | PyTorch, scikit-learn for metrics |
| Augmentation | Albumentations (both 1.x and 2.x constructor signatures supported) |
| Dataset access | `kagglehub`, `huggingface_hub` (PlantWild is on HF, not Kaggle) |
| Export | `onnx` is a dependency, but serving loads `best.pt` — the ONNX file stays with the training artifacts |
| Sources | PlantVillage, PlantDoc, PlantWild, PlantSeg, three lettuce sets |
| Domains | `studio` 22,200 / `field` 4,521 / `vertical` 11,347 images |

`v3` was warm-started from `v2` (`init_from`), trained 3 head epochs + 17
fine-tune epochs at batch 64, label smoothing 0.1, unfreezing from block 7.
The split is **grouped by perceptual hash**, not random, because the sources
are full of near-duplicates (43% in the field domain) and a random split would
fit the confidence threshold against re-photographs of the training set.

`ml/contract.py` is the single declaration of everything training and serving
must agree on. It exists because five separate breakages once traced back to
nothing declaring the contract; `backend/tests/test_contract.py` fails if the
two sides drift apart again.

## How a diagnosis flows through the stack

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
                                       ml.predict (imports ml.data)
                                                │
                                    Firestore status=completed|uncertain
                                                │
GET /diagnoses/{id}  or  /stream  <─────────────┘
```

## Deployment

One image for both the API and the worker — they differ only by command, and
sharing the image guarantees the worker's preprocessing matches what the API
validated against. Build context is the **repo root**, because the image needs
`ml/` alongside `backend/`:

```bash
docker compose -f backend/docker-compose.yml up --build
```

Torch installs as its own early layer from the CUDA index (`cu124`); swap to
`.../whl/cpu` for a CPU-only deployment and save ~2 GB. `opencv-python-headless`
still needs `libgl1` and `libglib2.0-0` at import time, which the image
installs. The model registry and secrets are **mounted, not baked in**, so a
retrained model is a file drop plus an activate call rather than an image
rebuild. GPU inference needs `nvidia-container-toolkit` and the commented-out
`deploy.resources` block in `docker-compose.yml`.

Installing torch from plain PyPI can silently give you a CPU-only build, which
trains roughly 50× slower on an RTX 4060. Both requirements files lead with
that warning; `start.sh` picks the right index automatically.

## Configuration

`backend/.env` (from `.env.example`), read by `app/config.py`:

| Variable | Purpose |
|---|---|
| `CORS_ORIGINS` | comma-separated; `*` is rejected |
| `FIREBASE_CREDENTIALS_PATH` | service-account JSON, or unset for workload identity |
| `FIREBASE_PROJECT_ID`, `FIREBASE_STORAGE_BUCKET` | project + bucket |
| `REDIS_URL` | broker and result backend |
| `MODEL_REGISTRY_DIR`, `DEFAULT_MODEL_VERSION`, `ML_REPO_ROOT` | model resolution |
| `INFERENCE_DEVICE` | unset => cuda if available, else cpu |
| `CELERY_TASK_TIME_LIMIT` / `_SOFT_TIME_LIMIT` | 120s / 100s |
| `MAX_UPLOAD_BYTES`, `MIN_IMAGE_DIMENSION` | 12 MB / 64px |

A blank `KEY=` line is treated as unset, and relative paths are anchored to
`backend/`. Both rules exist because a blank `MODEL_REGISTRY_DIR=` once
resolved to `.` — whatever directory the process happened to start in — and
silently broke the worker's `import ml.predict`.

`frontend/.env.local` carries `NEXT_PUBLIC_API_BASE_URL`,
`NEXT_PUBLIC_API_PREFIX` and the six `NEXT_PUBLIC_FIREBASE_*` values. That web
app **must** belong to the same Firebase project as the backend's service
account, or the API rejects every token it is handed.

## Testing and tooling

```bash
python -m pytest backend/tests/ -q     # pytest, pytest-asyncio, httpx
cd frontend && npm run lint            # eslint 9 + eslint-config-next
cd frontend && npm run typecheck       # tsc --noEmit
cd frontend && npm run format          # prettier + prettier-plugin-tailwindcss
```

The backend suite covers upload validation (including the truncated JPEG that
`Image.verify()` misses), registry resolution, the training/serving contract,
threshold calibration, dataset sources, split integrity, and the fork
safety rule above. Firebase and torch are not exercised — those need real
credentials and real weights.

`python -m ml.verify --download` is the pre-flight for training: it exercises
dataset discovery, splits, transforms and one forward+backward step, so a
long run does not die on a path typo.

## Deliberately not in the stack

| Not used | Why |
|---|---|
| A relational database / ORM | Firestore is already the auth and storage provider; a second datastore buys consistency problems before it buys queries. The cost is real: `/admin/stats` is a 501 because Firestore has no `GROUP BY`. |
| An auth server or session store | The Firebase client SDK owns the credential lifecycle; the API verifies and nothing more. |
| ONNX Runtime / TorchServe | Serving loads `best.pt` through the same `predict.py` training uses. A second runtime is a second chance for preprocessing to drift. |
| Model weights in git | ~9 MB per version, permanently, per retrain. They ship as verified release bundles instead. |
| A CDN or image pipeline | Images are authenticated per-user; they are fetched as blobs through the API. |
| Websockets | One-directional progress updates are what SSE is for. |
