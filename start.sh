#!/usr/bin/env bash
#
# Agrivert — one-command bootstrap and run script (backend + frontend).
#
#   ./start.sh              set up whatever is missing, then run everything
#   ./start.sh setup        set up only, don't run
#   ./start.sh doctor       report what is installed/configured, change nothing
#   ./start.sh stop         kill services left over from a previous run
#   ./start.sh clean        remove venv / node_modules / .next / logs
#   ./start.sh --help       full option list
#
# It is idempotent: safe to run on a clean checkout and safe to re-run.
# Everything it installs is confined to the repo (.venv, node_modules) except
# system packages (python/node/redis), which it installs via brew/apt only
# when they are missing.
#
# Deliberately NOT done automatically:
#   * filling in .env credentials — it copies the examples and tells you what
#     is still blank; Firebase keys cannot be guessed.
#   * scripts/seed_diseases.py — it writes to a real Firestore. Pass --seed.

set -eo pipefail

# ---------------------------------------------------------------- constants --

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
RUN_DIR="$ROOT/.run"
LOG_DIR="$RUN_DIR/logs"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"   # wheel index tag used with --cuda
MIN_NODE_MAJOR=20

# Defaults, overridable by flags (see usage()).
CMD="up"
DO_BACKEND=1
DO_FRONTEND=1
DO_INSTALL=1
DO_MODEL=1
DO_SEED=0
FRESH=0
AUTO_INSTALL=1
FORCE_CUDA=0
REDIS_MODE="auto"                   # auto | local | docker | none
DEV_MODEL_THRESHOLD="0.95"

if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'
  MAG=$'\033[35m'; CYN=$'\033[36m'; DIM=$'\033[2m'; BLD=$'\033[1m'; NC=$'\033[0m'
else
  RED=; GRN=; YLW=; BLU=; MAG=; CYN=; DIM=; BLD=; NC=
fi

step() { printf '\n%s==>%s %s%s%s\n' "$BLU" "$NC" "$BLD" "$*" "$NC"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s✓%s %s\n' "$GRN" "$NC" "$*"; }
warn() { printf '    %s!%s %s\n' "$YLW" "$NC" "$*"; }
err()  { printf '    %s✗%s %s\n' "$RED" "$NC" "$*" >&2; }
die()  { err "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'EOF'
Agrivert dev bootstrap.

USAGE
  ./start.sh [command] [options]

COMMANDS
  up          (default) set up anything missing, then run redis + celery +
              uvicorn + next dev, streaming all logs. Ctrl-C stops everything.
  setup       set up only; do not start any service.
  doctor      report environment/config status; change nothing.
  stop        stop services recorded by a previous run.
  clean       delete .venv, node_modules, .next, and .run/ (then re-run setup).

OPTIONS
  --backend-only        skip the frontend entirely
  --frontend-only       skip the backend entirely
  --fresh               rebuild deps from scratch (deletes .venv/node_modules)
  --skip-install        do not touch pip/npm (assume deps are installed)
  --cuda                install torch from the CUDA wheel index (Linux+NVIDIA)
                        tag from $TORCH_CUDA, default cu124
  --redis MODE          auto (default) | local | docker | none
  --seed                also run scripts/seed_diseases.py (writes to Firestore)
  --no-model            do not create a placeholder model when none exists
  --model-threshold X   confidence threshold for the placeholder model.
                        Use 0 for frontend work: the real 0.95 threshold makes
                        an untrained model return "uncertain" every time.
  --no-auto-install     never install system packages via brew/apt
  --api-port N          default 8000  (env: API_PORT)
  --web-port N          default 3000  (env: WEB_PORT)
  --python PATH         interpreter to build the venv with
  -h, --help            this text

EXAMPLES
  ./start.sh                         # first run: installs everything, starts all
  ./start.sh setup --fresh           # nuke and rebuild both dependency trees
  ./start.sh --frontend-only         # just next dev
  ./start.sh --model-threshold 0     # placeholder model that returns verdicts
  ./start.sh doctor                  # "why is my health endpoint degraded?"
EOF
}

# ------------------------------------------------------------ arg parsing ----

PYTHON_BIN="${PYTHON:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    up|setup|doctor|stop|clean) CMD="$1" ;;
    --backend-only)     DO_FRONTEND=0 ;;
    --frontend-only)    DO_BACKEND=0 ;;
    --fresh)            FRESH=1 ;;
    --skip-install)     DO_INSTALL=0 ;;
    --cuda)             FORCE_CUDA=1 ;;
    --redis)            REDIS_MODE="$2"; shift ;;
    --redis=*)          REDIS_MODE="${1#*=}" ;;
    --seed)             DO_SEED=1 ;;
    --no-model)         DO_MODEL=0 ;;
    --model-threshold)  DEV_MODEL_THRESHOLD="$2"; shift ;;
    --model-threshold=*) DEV_MODEL_THRESHOLD="${1#*=}" ;;
    --no-auto-install)  AUTO_INSTALL=0 ;;
    --api-port)         API_PORT="$2"; shift ;;
    --api-port=*)       API_PORT="${1#*=}" ;;
    --web-port)         WEB_PORT="$2"; shift ;;
    --web-port=*)       WEB_PORT="${1#*=}" ;;
    --python)           PYTHON_BIN="$2"; shift ;;
    --python=*)         PYTHON_BIN="${1#*=}" ;;
    -h|--help)          usage; exit 0 ;;
    *)                  usage; die "unknown argument: $1" ;;
  esac
  shift
done

case "$REDIS_MODE" in auto|local|docker|none) ;; *) die "--redis must be auto|local|docker|none" ;; esac

VENV="$BACKEND/.venv"
VENV_PY="$VENV/bin/python"
VENV_BIN="$VENV/bin"
USE_UV=0          # set by ensure_venv: uv if installed, pip otherwise

# --------------------------------------------------------------- utilities ---

# env_get <file> <KEY> — last assignment of KEY, unquoted. Comments ignored.
env_get() {
  [ -f "$1" ] || return 0
  sed -n "s/^[[:space:]]*$2[[:space:]]*=//p" "$1" | tail -n1 \
    | sed -e 's/^["'\'']//' -e 's/["'\'']$//' -e 's/[[:space:]]*$//' | tr -d '\r'
}

port_busy() {
  if have lsof; then
    [ -n "$(lsof -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null)" ]
  elif have nc; then
    nc -z 127.0.0.1 "$1" >/dev/null 2>&1
  else
    return 1
  fi
}

port_owner() { have lsof && lsof -iTCP:"$1" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | head -n1; }

# sys_install <brew-pkg> <apt-pkg> <what> — install a system package, but only
# if we're allowed to. The two package names differ often enough (node/nodejs,
# redis/redis-server) that guessing one from the other would be wrong.
sys_install() {
  local brew_pkg="$1" apt_pkg="$2" why="$3" sudo_cmd=""
  if [ "$AUTO_INSTALL" -eq 0 ]; then
    die "$why is missing and --no-auto-install was passed. Install it yourself and re-run."
  fi
  if have brew && [ -n "$brew_pkg" ]; then
    info "installing $brew_pkg with homebrew…"
    brew install "$brew_pkg"
  elif have apt-get && [ -n "$apt_pkg" ]; then
    [ "$(id -u)" -ne 0 ] && sudo_cmd="sudo"
    info "installing $apt_pkg with apt-get (may prompt for sudo)…"
    $sudo_cmd apt-get update -qq
    $sudo_cmd apt-get install -y "$apt_pkg"
  else
    die "$why is missing and no supported package manager (brew/apt) was found. Install it manually."
  fi
}

# wait_for_exit <pid-list> <half-seconds> — true once every pid is gone.
wait_for_exit() {
  local pids="$1" tries="$2" i=0 p alive
  while [ "$i" -lt "$tries" ]; do
    alive=0
    for p in $pids; do
      if kill -0 "$p" 2>/dev/null; then alive=1; fi
    done
    if [ "$alive" -eq 0 ]; then return 0; fi
    sleep 0.5
    i=$((i + 1))
  done
  return 1
}

# Recursively terminate a process and its children.
kill_tree() {
  local pid="$1" sig="${2:-TERM}" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
  kill -"$sig" "$pid" 2>/dev/null || true
}

wait_http() {  # wait_http <url> <seconds>
  local url="$1" tries="${2:-60}" i=0
  while [ "$i" -lt "$tries" ]; do
    curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null && return 0
    sleep 1; i=$((i + 1))
  done
  return 1
}

# --------------------------------------------------------- redis discovery ---

redis_url() {
  local u; u="$(env_get "$BACKEND/.env" REDIS_URL)"
  echo "${u:-redis://localhost:6379/0}"
}

parse_redis() {
  local hp; hp="$(redis_url)"
  hp="${hp#*://}"; hp="${hp%%/*}"; hp="${hp##*@}"
  REDIS_HOST="${hp%%:*}"
  REDIS_PORT="${hp##*:}"
  [ "$REDIS_PORT" = "$REDIS_HOST" ] && REDIS_PORT=6379
  [ -n "$REDIS_HOST" ] || REDIS_HOST=localhost
}

redis_alive() {
  parse_redis
  if have redis-cli; then
    [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null)" = "PONG" ]
  elif have nc; then
    nc -z "$REDIS_HOST" "$REDIS_PORT" >/dev/null 2>&1
  else
    port_busy "$REDIS_PORT"
  fi
}

# ------------------------------------------------------------ python setup ---

pick_python() {
  local candidate
  # An existing venv wins, so re-runs never silently switch interpreter.
  if [ "$FRESH" -eq 0 ] && [ -x "$VENV_PY" ]; then
    PYTHON_BIN="$VENV_PY"; return 0
  fi
  if [ -n "$PYTHON_BIN" ]; then
    have "$PYTHON_BIN" || [ -x "$PYTHON_BIN" ] || die "--python $PYTHON_BIN not found"
    return 0
  fi
  for candidate in python3.12 python3.11 python3.13 python3; do
    if have "$candidate" && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"; return 0
    fi
  done
  sys_install "python@3.12" "python3" "python >= 3.10"
  have python3.12 && PYTHON_BIN=python3.12 || PYTHON_BIN=python3
}

ensure_venv() {
  if [ "$FRESH" -eq 1 ] && [ -d "$VENV" ]; then
    info "removing $VENV (--fresh)"
    rm -rf "$VENV"
  fi
  if [ ! -x "$VENV_PY" ]; then
    if have uv; then
      info "creating venv with uv"
      uv venv --python "${UV_PYTHON:-3.12}" "$VENV" >/dev/null
    else
      pick_python
      info "creating venv with $("$PYTHON_BIN" --version 2>&1)"
      "$PYTHON_BIN" -m venv "$VENV" || {
        # Debian splits venv into its own package.
        have apt-get && sys_install "" "python3-venv" "the python venv module"
        "$PYTHON_BIN" -m venv "$VENV" || die "could not create a venv with $PYTHON_BIN"
      }
    fi
  fi

  # uv-created venvs have no pip at all, and a `python -m venv` on a Debian
  # box can be missing it too. Prefer uv when it's here; otherwise bootstrap
  # pip so the same code path works either way.
  if have uv; then
    USE_UV=1
  elif ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    info "venv has no pip — bootstrapping with ensurepip"
    "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 \
      || die "no pip in $VENV and ensurepip failed. Install uv, or recreate the venv with --fresh."
  fi
  ok "venv: $("$VENV_PY" --version 2>&1)$( [ "$USE_UV" -eq 1 ] && echo " (installing with uv)" )"
}

# Install into the venv, through uv when available and pip otherwise.
py_install() {
  if [ "$USE_UV" -eq 1 ]; then
    uv pip install --python "$VENV_PY" "$@"
  else
    "$VENV_PY" -m pip install --disable-pip-version-check "$@"
  fi
}

install_torch() {
  if [ "$FRESH" -eq 0 ] && "$VENV_PY" -c 'import torch, torchvision' >/dev/null 2>&1; then
    ok "torch $("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null) already installed"
    return 0
  fi
  # torch must be installed before requirements.txt: plain PyPI can hand you a
  # CPU-only build on Linux, and requirements.txt would then consider the
  # constraint satisfied. Same caveat as backend/requirements.txt's header.
  local index=""
  if [ "$FORCE_CUDA" -eq 1 ]; then
    index="https://download.pytorch.org/whl/$TORCH_CUDA"
  elif [ "$(uname -s)" = "Darwin" ]; then
    index=""                                     # macOS wheels are MPS/CPU, PyPI is correct
  elif have nvidia-smi; then
    index="https://download.pytorch.org/whl/$TORCH_CUDA"
    info "NVIDIA GPU detected — using the $TORCH_CUDA wheel index"
  else
    index="https://download.pytorch.org/whl/cpu"
    info "no GPU detected — installing the CPU build of torch"
  fi

  if [ -n "$index" ]; then
    py_install torch torchvision --index-url "$index"
  else
    py_install torch torchvision
  fi
}

backend_env_file() {
  if [ ! -f "$BACKEND/.env" ]; then
    cp "$BACKEND/.env.example" "$BACKEND/.env"
    warn "created backend/.env from the example — fill in FIREBASE_* before uploads will work"
  fi
  mkdir -p "$BACKEND/secrets"
  local cred; cred="$(env_get "$BACKEND/.env" FIREBASE_CREDENTIALS_PATH)"
  cred="${cred:-./secrets/serviceAccount.json}"
  case "$cred" in /*) : ;; *) cred="$BACKEND/${cred#./}" ;; esac
  if [ ! -f "$cred" ]; then
    warn "no service account at $cred"
    info "  Firebase console > Project settings > Service accounts > Generate new private key"
    info "  Everything except /health and the model registry will fail without it."
  else
    ok "service account present"
  fi
  local proj; proj="$(env_get "$BACKEND/.env" FIREBASE_PROJECT_ID)"
  case "$proj" in ""|your-project-id) warn "FIREBASE_PROJECT_ID is still a placeholder in backend/.env" ;; esac
}

model_registry_dir() {
  local d; d="$(env_get "$BACKEND/.env" MODEL_REGISTRY_DIR)"
  d="${d:-./models}"
  case "$d" in /*) echo "$d" ;; *) echo "$BACKEND/${d#./}" ;; esac
}

count_models() {
  local dir n=0 d
  dir="$(model_registry_dir)"
  [ -d "$dir" ] || { echo 0; return; }
  for d in "$dir"/*/; do
    [ -d "$d" ] || continue
    if [ -f "$d/best.pt" ] && [ -f "$d/metadata.json" ] && [ -f "$d/labels.json" ]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

ensure_model() {
  local n; n="$(count_models)"
  if [ "$n" -gt 0 ]; then
    ok "$n model version(s) in $(model_registry_dir)"
    if [ "$n" -gt 1 ] && [ -z "$(env_get "$BACKEND/.env" DEFAULT_MODEL_VERSION)" ]; then
      warn "several versions and no DEFAULT_MODEL_VERSION — the registry will refuse to guess"
      info "  set DEFAULT_MODEL_VERSION in backend/.env, or activate one via POST /admin/models/{v}/activate"
    fi
    return 0
  fi
  if [ "$DO_MODEL" -eq 0 ]; then
    warn "no model version on disk and --no-model was passed; inference will fail"
    return 0
  fi
  warn "no model version on disk — generating an UNTRAINED placeholder so the pipeline can run"
  ( cd "$BACKEND" && "$VENV_PY" scripts/bootstrap_dev_model.py --threshold "$DEV_MODEL_THRESHOLD" )
  warn "that model's weights are random; every verdict is noise. Replace it with:"
  info "  backend/.venv/bin/python scripts/register_model.py <artifacts-dir> --version v1-… --activate"
}

setup_backend() {
  step "Backend"
  backend_env_file
  ensure_venv
  if [ "$DO_INSTALL" -eq 1 ]; then
    [ "$USE_UV" -eq 0 ] && py_install -q --upgrade pip wheel
    install_torch
    info "installing backend/requirements.txt…"
    py_install -q -r "$BACKEND/requirements.txt"
    ok "python dependencies installed"
  else
    info "skipping pip (--skip-install)"
  fi
  ensure_model
  if [ "$DO_SEED" -eq 1 ]; then
    info "seeding the disease knowledge base…"
    ( cd "$BACKEND" && "$VENV_PY" scripts/seed_diseases.py )
  fi
}

# ---------------------------------------------------------- frontend setup ---

ensure_node() {
  if ! have node; then
    sys_install node nodejs "node >= $MIN_NODE_MAJOR"
  fi
  local major; major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  if [ "$major" -lt "$MIN_NODE_MAJOR" ]; then
    die "node $(node --version) is too old; Next 16 needs >= $MIN_NODE_MAJOR. Upgrade node and re-run."
  fi
  have npm || die "npm not found alongside node $(node --version)"
  ok "node $(node --version), npm $(npm --version)"
}

frontend_env_file() {
  if [ ! -f "$FRONTEND/.env.local" ] && [ ! -f "$FRONTEND/.env" ]; then
    cp "$FRONTEND/.env.example" "$FRONTEND/.env.local"
    warn "created frontend/.env.local from the example — fill in the Firebase web config"
  fi
  local envf="$FRONTEND/.env.local"
  [ -f "$envf" ] || envf="$FRONTEND/.env"
  local key; key="$(env_get "$envf" NEXT_PUBLIC_FIREBASE_API_KEY)"
  if [ -z "$key" ]; then
    warn "NEXT_PUBLIC_FIREBASE_API_KEY is empty in $(basename "$envf") — login will not work"
    info "  Firebase console > Project settings > Your apps > Web app > SDK setup"
  else
    ok "firebase web config present"
  fi
  # The browser talks to the API directly, so a port change has to be mirrored.
  local base; base="$(env_get "$envf" NEXT_PUBLIC_API_BASE_URL)"
  case "$base" in
    *":$API_PORT"*|"") : ;;
    *) warn "NEXT_PUBLIC_API_BASE_URL is $base but the API will run on port $API_PORT" ;;
  esac
}

setup_frontend() {
  step "Frontend"
  ensure_node
  frontend_env_file
  if [ "$DO_INSTALL" -eq 0 ]; then
    info "skipping npm (--skip-install)"
    return 0
  fi
  if [ "$FRESH" -eq 1 ] && [ -d "$FRONTEND/node_modules" ]; then
    info "removing node_modules (--fresh)"
    rm -rf "$FRONTEND/node_modules"
  fi
  if [ ! -d "$FRONTEND/node_modules" ]; then
    if [ -f "$FRONTEND/package-lock.json" ]; then
      info "npm ci…"
      ( cd "$FRONTEND" && npm ci --no-audit --no-fund )
    else
      info "npm install…"
      ( cd "$FRONTEND" && npm install --no-audit --no-fund )
    fi
  elif [ "$FRONTEND/package-lock.json" -nt "$FRONTEND/node_modules/.package-lock.json" ]; then
    # npm stamps node_modules/.package-lock.json on every install, so this
    # compares against the last install rather than the directory's mtime.
    info "lockfile is newer than node_modules — npm install…"
    ( cd "$FRONTEND" && npm install --no-audit --no-fund )
  fi
  ok "node dependencies installed"
}

# ------------------------------------------------------------- run services --

SERVICE_PIDS=""
SERVICE_LIST=""     # "name:pid name:pid …"
TAIL_PIDS=""
STARTED_DOCKER_REDIS=0
SHUTTING_DOWN=0

stream_log() {  # stream_log <name> <color> <file>
  local name="$1" color="$2" file="$3"
  : > "$file"
  # stderr of the whole subshell is dropped so that killing it at shutdown
  # doesn't print a job-control "Terminated: 15" line per service.
  (
    tail -n 0 -F "$file" 2>/dev/null | while IFS= read -r line; do
      printf '%s%-6s%s %s│%s %s\n' "$color" "$name" "$NC" "$DIM" "$NC" "$line"
    done
  ) 2>/dev/null &
  local pid=$!
  TAIL_PIDS="$TAIL_PIDS $pid"
  disown "$pid" 2>/dev/null || true   # keeps "Terminated: 15" out of the output
}

start_service() {  # start_service <name> <color> <workdir> <cmd…>
  local name="$1" color="$2" dir="$3"; shift 3
  local log="$LOG_DIR/$name.log"
  stream_log "$name" "$color" "$log"
  ( cd "$dir" && exec "$@" ) >>"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_DIR/$name.pid"
  SERVICE_PIDS="$SERVICE_PIDS $pid"
  SERVICE_LIST="$SERVICE_LIST $name:$pid"
  disown "$pid" 2>/dev/null || true
  info "$name started (pid $pid, log $log)"
}

start_redis() {
  [ "$REDIS_MODE" = "none" ] && { info "redis: not managed (--redis none)"; return 0; }
  parse_redis
  if redis_alive; then
    ok "redis already running at $REDIS_HOST:$REDIS_PORT — reusing it"
    return 0
  fi
  case "$REDIS_HOST" in
    localhost|127.0.0.1|::1) : ;;
    *) die "REDIS_URL points at $REDIS_HOST, which is not reachable and not ours to start" ;;
  esac

  local mode="$REDIS_MODE"
  if [ "$mode" = "auto" ]; then
    if have redis-server; then mode=local
    elif have docker;      then mode=docker
    else
      sys_install redis redis-server "redis"
      mode=local
    fi
  fi

  if [ "$mode" = "local" ]; then
    have redis-server || sys_install redis redis-server "redis-server"
    start_service redis "$YLW" "$ROOT" redis-server --port "$REDIS_PORT" --save '' --appendonly no
  else
    have docker || die "--redis docker was requested but docker is not installed"
    info "starting redis in docker (container: agrivert-redis)…"
    docker rm -f agrivert-redis >/dev/null 2>&1 || true
    docker run -d --rm --name agrivert-redis -p "$REDIS_PORT:6379" redis:7-alpine >/dev/null
    STARTED_DOCKER_REDIS=1
  fi

  local i=0
  while [ "$i" -lt 20 ]; do
    redis_alive && { ok "redis up on $REDIS_PORT"; return 0; }
    sleep 0.5; i=$((i + 1))
  done
  die "redis did not come up on port $REDIS_PORT"
}

start_backend_services() {
  [ -x "$VENV_PY" ] || die "backend venv missing — run ./start.sh setup first"
  if port_busy "$API_PORT"; then
    die "port $API_PORT is already in use$( [ -n "$(port_owner "$API_PORT")" ] && echo " (by $(port_owner "$API_PORT"))" ). Try --api-port, or ./start.sh stop."
  fi
  export PYTHONUNBUFFERED=1
  # prefork is safe here only because app/worker/tasks.py keeps grpc out of
  # the parent process — see the fork-safety note at the top of that file
  # before adding imports to it.
  start_service worker "$MAG" "$BACKEND" \
    "$VENV_BIN/celery" -A app.worker.celery_app:celery_app worker --loglevel=info --concurrency=1
  start_service api "$CYN" "$BACKEND" \
    "$VENV_BIN/uvicorn" app.main:app --reload --host 0.0.0.0 --port "$API_PORT"
}

start_frontend_service() {
  [ -d "$FRONTEND/node_modules" ] || die "frontend deps missing — run ./start.sh setup first"
  if port_busy "$WEB_PORT"; then
    die "port $WEB_PORT is already in use. Try --web-port, or ./start.sh stop."
  fi
  start_service web "$GRN" "$FRONTEND" npm run dev -- --port "$WEB_PORT"
}

report_health() {
  local url="http://127.0.0.1:$API_PORT$(env_get "$BACKEND/.env" API_PREFIX)"
  url="${url%/}/health"
  if ! wait_http "$url" 90; then
    warn "API did not answer $url in time — check $LOG_DIR/api.log"
    return 0
  fi
  local body; body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || echo '')"
  # Small enough to read with the venv's python; jq may not be installed.
  # /health responds camelCase (schemas.common.CamelModel).
  "$VENV_PY" - "$body" "$GRN" "$RED" "$NC" <<'PY' 2>/dev/null || info "health: $body"
import json, sys

body, green, red, nc = sys.argv[1:5]
try:
    d = json.loads(body)
except Exception:
    raise SystemExit(1)

get = lambda k: d.get(k[0], d.get(k[1]))
mark = lambda b: (f"{green}✓{nc}" if b else f"{red}✗{nc}")
print(f"    health: {d.get('status')}   "
      f"{mark(get(('modelReady', 'model_ready')))} model({get(('modelVersion', 'model_version'))})  "
      f"{mark(get(('firestoreReady', 'firestore_ready')))} firestore  "
      f"{mark(get(('brokerReady', 'broker_ready')))} broker")

# Firestore/GCP errors arrive as a multi-line protobuf dump; one line is
# enough to identify the problem, and the full text is in the api log.
detail = (d.get("detail") or "").strip()
if detail:
    first = detail.splitlines()[0]
    truncated = len(first) > 160 or len(detail.splitlines()) > 1
    print("    detail: " + first[:160] + ("…" if truncated else ""))
    print("            (full error in .run/logs/api.log)")
PY
}

cleanup() {
  # errexit off for the whole handler: a `kill -0` on an already-dead pid is
  # expected here, and under `set -e` that aborts the shutdown halfway and
  # leaves orphaned services behind.
  set +e
  if [ "$SHUTTING_DOWN" -eq 1 ]; then return 0; fi
  SHUTTING_DOWN=1
  trap - INT TERM EXIT
  printf '\n%s==>%s shutting down…\n' "$BLU" "$NC"

  local p
  # TERM the service leader only, never its children first: celery's prefork
  # parent immediately respawns a child killed underneath it.
  for p in $SERVICE_PIDS; do kill -TERM "$p" 2>/dev/null; done

  # A second TERM is celery's *cold* shutdown; without it an idle worker can
  # sit in warm shutdown for ten seconds. Harmless for the other three.
  if ! wait_for_exit "$SERVICE_PIDS" 8; then
    for p in $SERVICE_PIDS; do kill -TERM "$p" 2>/dev/null; done
  fi
  # Anything still up after that gets its whole tree killed.
  if ! wait_for_exit "$SERVICE_PIDS" 8; then
    for p in $SERVICE_PIDS; do
      if kill -0 "$p" 2>/dev/null; then kill_tree "$p" KILL; fi
    done
  fi
  for p in $TAIL_PIDS; do kill_tree "$p" TERM; done

  if [ "$STARTED_DOCKER_REDIS" -eq 1 ]; then
    docker rm -f agrivert-redis >/dev/null 2>&1
  fi
  rm -f "$RUN_DIR"/*.pid 2>/dev/null
  printf '    %sstopped%s\n' "$GRN" "$NC"
}

run_all() {
  mkdir -p "$LOG_DIR"
  # On a signal, exit from inside the handler. Returning from it would drop
  # back into the watchdog loop below, which would then report the services
  # we just stopped as crashes.
  trap 'cleanup; exit 0' INT TERM
  trap cleanup EXIT

  step "Starting services"
  if [ "$DO_BACKEND" -eq 1 ]; then
    start_redis
    start_backend_services
  fi
  if [ "$DO_FRONTEND" -eq 1 ]; then
    start_frontend_service
  fi

  if [ "$DO_BACKEND" -eq 1 ]; then
    step "Waiting for the API"
    report_health
  fi
  if [ "$DO_FRONTEND" -eq 1 ]; then
    if wait_http "http://127.0.0.1:$WEB_PORT" 120; then
      ok "web up"
    else
      warn "web did not answer yet — check $LOG_DIR/web.log"
    fi
  fi

  step "Ready"
  if [ "$DO_BACKEND" -eq 1 ]; then
    info "API      http://localhost:$API_PORT"
    info "Docs     http://localhost:$API_PORT/docs"
  fi
  if [ "$DO_FRONTEND" -eq 1 ]; then
    info "Web      http://localhost:$WEB_PORT"
  fi
  info "Logs     $LOG_DIR"
  printf '    %sCtrl-C stops everything.%s\n\n' "$DIM" "$NC"

  # Surface a crashed service instead of leaving a half-dead stack running.
  local entry name pid
  while :; do
    for entry in $SERVICE_LIST; do
      name="${entry%%:*}"; pid="${entry#*:}"
      if ! kill -0 "$pid" 2>/dev/null; then
        # `./start.sh stop` in another terminal deletes the pid files, so a
        # missing one means "someone stopped us", not "the service crashed".
        if [ ! -f "$RUN_DIR/$name.pid" ]; then
          info "$name was stopped externally — shutting the rest down"
          exit 0
        fi
        err "$name exited unexpectedly — last lines of $LOG_DIR/$name.log:"
        tail -n 20 "$LOG_DIR/$name.log" 2>/dev/null | sed 's/^/      /'
        exit 1
      fi
    done
    sleep 2
  done
}

# ------------------------------------------------------------- subcommands ---

cmd_stop() {
  local f name pid found=0 pids=""
  for f in "$RUN_DIR"/*.pid; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .pid)"; pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      info "stopping $name (pid $pid)"
      kill -TERM "$pid" 2>/dev/null
      pids="$pids $pid"
      found=1
    fi
    rm -f "$f"
  done
  if [ -n "$pids" ] && ! wait_for_exit "$pids" 16; then
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then kill_tree "$pid" KILL; fi
    done
  fi
  if have docker && [ -n "$(docker ps -q -f name=agrivert-redis 2>/dev/null)" ]; then
    info "removing docker container agrivert-redis"
    docker rm -f agrivert-redis >/dev/null 2>&1 || true
    found=1
  fi
  [ "$found" -eq 1 ] && ok "stopped" || info "nothing was running"
}

cmd_clean() {
  step "Cleaning"
  cmd_stop
  for target in "$VENV" "$FRONTEND/node_modules" "$FRONTEND/.next" "$RUN_DIR"; do
    if [ -e "$target" ]; then
      info "removing ${target#$ROOT/}"
      rm -rf "$target"
    fi
  done
  ok "clean. Run ./start.sh to rebuild."
}

cmd_doctor() {
  step "Tooling"
  have python3 && ok "python3 $(python3 --version 2>&1 | cut -d' ' -f2)" || warn "python3 not installed"
  [ -x "$VENV_PY" ] && ok "venv $("$VENV_PY" --version 2>&1 | cut -d' ' -f2) at backend/.venv" || warn "no backend venv"
  have node && ok "node $(node --version)" || warn "node not installed"
  have redis-server && ok "redis-server $(redis-server --version 2>/dev/null | sed -n 's/.*v=\([^ ]*\).*/\1/p')" || warn "redis-server not installed"
  have docker && ok "docker present" || info "docker not installed (fine unless you use --redis docker)"

  step "Backend"
  if [ -x "$VENV_PY" ]; then
    "$VENV_PY" -c 'import torch' 2>/dev/null \
      && ok "torch $("$VENV_PY" -c 'import torch;print(torch.__version__)')" || warn "torch not installed in the venv"
    "$VENV_PY" -c 'import fastapi, celery, firebase_admin' 2>/dev/null \
      && ok "fastapi / celery / firebase-admin importable" || warn "backend requirements not fully installed"
  fi
  [ -f "$BACKEND/.env" ] && ok "backend/.env present" || warn "backend/.env missing"
  local cred; cred="$(env_get "$BACKEND/.env" FIREBASE_CREDENTIALS_PATH)"; cred="${cred:-./secrets/serviceAccount.json}"
  case "$cred" in /*) : ;; *) cred="$BACKEND/${cred#./}" ;; esac
  [ -f "$cred" ] && ok "service account at ${cred#$ROOT/}" || warn "no service account at ${cred#$ROOT/}"
  info "model versions on disk: $(count_models) in $(model_registry_dir)"
  redis_alive && ok "redis reachable at $(redis_url)" || warn "redis not reachable at $(redis_url)"
  port_busy "$API_PORT" && warn "port $API_PORT is in use ($(port_owner "$API_PORT"))" || ok "port $API_PORT free"

  step "Frontend"
  [ -d "$FRONTEND/node_modules" ] && ok "node_modules present" || warn "node_modules missing"
  { [ -f "$FRONTEND/.env.local" ] || [ -f "$FRONTEND/.env" ]; } \
    && ok "frontend env file present" || warn "no frontend .env.local"
  port_busy "$WEB_PORT" && warn "port $WEB_PORT is in use ($(port_owner "$WEB_PORT"))" || ok "port $WEB_PORT free"
  echo
}

# ------------------------------------------------------------------- main ----

[ -d "$BACKEND" ] || die "run this from the agrivert repo (backend/ not found next to $0)"

case "$CMD" in
  stop)   step "Stopping"; cmd_stop ;;
  clean)  cmd_clean ;;
  doctor) cmd_doctor ;;
  setup|up)
    mkdir -p "$LOG_DIR"
    if [ "$DO_BACKEND" -eq 1 ];  then setup_backend;  fi
    if [ "$DO_FRONTEND" -eq 1 ]; then setup_frontend; fi
    if [ "$CMD" = "setup" ]; then
      step "Setup complete"
      info "start everything with: ./start.sh"
    else
      run_all
    fi
    ;;
esac
