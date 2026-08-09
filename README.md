# Agrivert

Plant disease diagnosis for growers. A photo goes up, a fine-tuned CNN
classifies it, and a verdict comes back with its uncertainty attached — or
comes back as `uncertain`, which happens often and on purpose. A confident
wrong answer to someone about to treat a crop costs more than an honest
"I can't tell".

```
ml/         training + the contract training and serving must agree on
backend/    FastAPI + Celery, imports ml/ directly
frontend/   Next.js client
```

## Quickstart

Two commands from a clean checkout:

```bash
./scripts/model-download.sh --activate    # install the trained model (~9 MB)
./start.sh                                # set up everything, then run it
```

`start.sh` streams redis, the celery worker, the API and `next dev` into one
terminal; Ctrl-C stops all four. It is idempotent, so it is also how you start
work on any later day. Logs land in `.run/logs/`.

The model download is separate because it is the one step that reaches the
network for something big, and because you may already have a model. Skip it
and `start.sh` generates an untrained placeholder instead — see below.

```bash
./start.sh doctor          # what's installed, what's configured, what's degraded
./start.sh setup           # prepare only, don't start anything
./start.sh --backend-only  # or --frontend-only
./start.sh --fresh         # rebuild .venv and node_modules from scratch
./start.sh stop            # kill services left behind by a previous run
./start.sh clean           # delete .venv, node_modules, .next, .run
./start.sh --help          # every flag
```

From a clean checkout `start.sh` will create `backend/.venv` (via `uv` if
installed), install the right torch build for the machine (CUDA wheels on an
NVIDIA Linux box, PyPI on macOS, CPU wheels otherwise), install
`requirements.txt` and run `npm ci`, copy both `.env.example` files, and
install any missing system packages (python, node, redis) via brew/apt.

Two things it deliberately will not do, because they cannot be guessed or are
not safely repeatable:

| Not done for you | Why |
|---|---|
| Filling in credentials | It copies both `.env.example` files and tells you which values are still blank. `backend/secrets/serviceAccount.json` has to come from the Firebase console. |
| Seeding Firestore | `backend/scripts/seed_diseases.py` writes to a real project. Pass `--seed` when you want it. |

## The model

**Do not train it.** Training needs a GPU, ~10 GB of datasets and hours.
One person does that; everyone else installs the result:

```bash
./scripts/model-download.sh              # install the pinned release
./scripts/model-download.sh --activate   # …and point backend/.env at it
```

It downloads the published bundle, verifies the archive hash and every file
hash inside it, and installs a version the backend resolves immediately. A
truncated download fails loudly instead of serving weights that load fine and
predict garbage. Which release it installs is pinned in one place,
`scripts/model-release.env` — nothing else repeats the URL and hash.

After training, the person who trained it publishes with:

```bash
./scripts/model-upload.sh artifacts/vertical --version v4-vertical-20260901
```

That packs the bundle, attaches all three files to a GitHub Release, and
repins `model-release.env` so everyone else's `model-download.sh` picks it up.
Full details, including what the current model is and is not good at, are in
[`ml/README.md`](ml/README.md).

### The placeholder model

If `backend/models/` is empty and no bundle has been installed, `start.sh`
runs `backend/scripts/bootstrap_dev_model.py`. Its weights are random — every
verdict is noise — and the version is named `v0-dev-untrained-*` so it can
never be mistaken for a trained one. At an honest confidence threshold an
untrained model returns `uncertain` every single time, so for frontend work on
the `completed` path:

```bash
./start.sh --model-threshold 0
```

## Manual operation

`start.sh` is a convenience, not a dependency. Everything it does can be run
by hand: `backend/README.md` has the three-terminal (or `docker compose up`)
workflow, `frontend/README.md` has `npm run dev`.

## Where to read next

| | |
|---|---|
| [`TECH_STACK.md`](TECH_STACK.md) | every technology in the project, its version, and why it is there |
| [`ROUTES.md`](ROUTES.md) | the API plan, and the gap in the original workflow each route closes |
| [`ml/README.md`](ml/README.md) | training, model distribution, and what the numbers actually mean |
| [`backend/README.md`](backend/README.md) | serving architecture, what is implemented and what returns 501 |
| [`frontend/README.md`](frontend/README.md) | the client, its auth model, and its design decisions |

`ml/` is **imported** by the backend, not copied into it. An earlier
arrangement kept vendored copies of `predict.py` and `data.py` under
`backend/`; they drifted, and a teammate's first run died on an `ImportError`
for names that had never existed. `ml/contract.py` now owns every constant
both sides need, and `backend/tests/test_contract.py` fails if they drift
apart again.
