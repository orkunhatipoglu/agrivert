# Agrivert

Plant disease diagnosis for farmers: a photo goes up, a fine-tuned CNN
classifies it, and a hedged verdict comes back. `ROUTES.md` is the API plan,
`backend/README.md` the serving architecture, `frontend/` the Next.js client.

## Quickstart

```bash
./start.sh
```

That one command sets up whatever is missing and then runs the whole stack —
redis, the celery worker, the API, and `next dev` — streaming all four logs
into one terminal. Ctrl-C stops everything. It is idempotent, so it is also
the normal way to start work on any later day.

From a clean checkout it will: create `backend/.venv` (via `uv` if installed,
`python -m venv` otherwise), install the right torch build for the machine
(CUDA wheels on an NVIDIA Linux box, PyPI on macOS, CPU wheels otherwise),
install `requirements.txt` and `npm ci`, copy both `.env.example` files, and —
if the model registry is empty — generate an **untrained placeholder model**
so the upload → queue → worker → poll path runs before a real checkpoint
exists. Missing system packages (python, node, redis) are installed via
brew/apt.

Two things it deliberately will not do, because they cannot be guessed or are
not safely repeatable:

- **Fill in credentials.** It copies `backend/.env.example` and
  `frontend/.env.example`, then tells you which values are still blank. The
  Firebase service account (`backend/secrets/serviceAccount.json`) has to be
  downloaded from the Firebase console.
- **Seed Firestore.** `scripts/seed_diseases.py` writes to a real project;
  pass `--seed` when you want it.

```bash
./start.sh doctor          # what's installed, what's configured, what's degraded
./start.sh setup           # prepare only, don't start anything
./start.sh --backend-only  # or --frontend-only
./start.sh --fresh         # rebuild .venv and node_modules from scratch
./start.sh stop            # kill services left behind by a previous run
./start.sh clean           # delete .venv, node_modules, .next, .run
./start.sh --help          # every flag
```

Service logs land in `.run/logs/`, pids in `.run/`.

### The placeholder model

If `backend/models/` is empty, `start.sh` runs
`scripts/bootstrap_dev_model.py`. Its weights are random — every verdict is
noise, and the version is named `v0-dev-untrained-*` so it cannot be mistaken
for a trained one. At the honest 0.95 confidence threshold an untrained model
returns `uncertain` every single time, so for frontend work on the
`completed` path:

```bash
./start.sh --model-threshold 0
```

Replace it with real weights as soon as you have them:

```bash
backend/.venv/bin/python backend/scripts/register_model.py <artifacts-dir> \
    --version v1-blended-20260808 --activate
```

## Manual operation

`start.sh` is a convenience, not a dependency — everything it does can be run
by hand. See `backend/README.md` for the three-terminal (or
`docker compose up`) workflow and `frontend/README.md` for `npm run dev`.
