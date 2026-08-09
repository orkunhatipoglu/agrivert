#!/usr/bin/env bash
#
# Install the published model bundle. This is how everyone who is not the
# person who trained it gets a working model: no GPU, no datasets, no training.
#
#   ./scripts/model-download.sh                 # install the pinned release
#   ./scripts/model-download.sh --activate      # …and serve it
#   ./scripts/model-download.sh --url <url> --sha256 <hash>
#
# Wraps `python -m ml.bundle fetch`, which verifies the archive hash and every
# file hash inside it before installing anything.

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ENV="$ROOT/scripts/model-release.env"

if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'
  DIM=$'\033[2m'; BLD=$'\033[1m'; NC=$'\033[0m'
else
  RED=; GRN=; YLW=; BLU=; DIM=; BLD=; NC=
fi
step() { printf '\n%s==>%s %s%s%s\n' "$BLU" "$NC" "$BLD" "$*" "$NC"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s✓%s %s\n' "$GRN" "$NC" "$*"; }
warn() { printf '    %s!%s %s\n' "$YLW" "$NC" "$*"; }
die()  { printf '    %s✗%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

INTO="$ROOT/backend/models"
ACTIVATE=0
FORCE=0
URL=""; SHA=""; VERSION=""

usage() {
  cat <<'EOF'
Install a model bundle into the backend registry.

USAGE
  ./scripts/model-download.sh [options]

OPTIONS
  --url URL         bundle .tar.gz  (default: pinned in scripts/model-release.env)
  --sha256 HASH     expected archive hash (default: pinned)
  --version NAME    override the installed directory name
  --into DIR        registry dir (default: backend/models)
  --activate        set DEFAULT_MODEL_VERSION in backend/.env to this version
  --force           replace the version if it is already installed
  -h, --help        this text

The URL and hash belong to the same release and are checked against each
other. Take both from the same release page, or leave both alone.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --url)        URL="$2"; shift ;;
    --url=*)      URL="${1#*=}" ;;
    --sha256)     SHA="$2"; shift ;;
    --sha256=*)   SHA="${1#*=}" ;;
    --version)    VERSION="$2"; shift ;;
    --version=*)  VERSION="${1#*=}" ;;
    --into)       INTO="$2"; shift ;;
    --into=*)     INTO="${1#*=}" ;;
    --activate)   ACTIVATE=1 ;;
    --force)      FORCE=1 ;;
    -h|--help)    usage; exit 0 ;;
    *)            usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

# Pinned release, unless the caller named one.
if [ -z "$URL" ]; then
  [ -f "$RELEASE_ENV" ] || die "no --url given and $RELEASE_ENV is missing"
  # shellcheck disable=SC1090
  . "$RELEASE_ENV"
  URL="$MODEL_URL"
  [ -n "$SHA" ] || SHA="$MODEL_SHA256"
  [ -n "$VERSION" ] || VERSION="$MODEL_VERSION"
  info "using the pinned release from scripts/model-release.env"
elif [ -z "$SHA" ]; then
  # Not fatal — ml.bundle prints the hash it saw — but an unverified bundle
  # can install weights that load fine and predict garbage.
  warn "--url without --sha256: the download will not be verified"
fi

PY="$ROOT/backend/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "no python found (run ./start.sh setup first)"

step "Installing ${VERSION:-bundle}"
info "from $URL"

FETCH_ARGS=(-m ml.bundle fetch "$URL" --into "$INTO")
if [ -n "$SHA" ];      then FETCH_ARGS+=(--sha256 "$SHA");    fi
if [ -n "$VERSION" ];  then FETCH_ARGS+=(--version "$VERSION"); fi
if [ "$FORCE" -eq 1 ]; then FETCH_ARGS+=(--force);            fi

# Run from the repo root so `ml` is importable without installing it.
( cd "$ROOT" && "$PY" "${FETCH_ARGS[@]}" ) || die "fetch failed — nothing was installed"

DEST="$INTO/${VERSION:-unknown}"
ok "installed $DEST"

if [ "$ACTIVATE" -eq 1 ]; then
  ENV_FILE="$ROOT/backend/.env"
  [ -n "$VERSION" ] || die "--activate needs a version name; pass --version"
  # On a clean checkout backend/.env does not exist yet, and requiring
  # ./start.sh setup first would make the order of the two quickstart
  # commands matter. Seed it from the example the same way start.sh does.
  if [ ! -f "$ENV_FILE" ]; then
    [ -f "$ROOT/backend/.env.example" ] || die "neither backend/.env nor backend/.env.example exists"
    cp "$ROOT/backend/.env.example" "$ENV_FILE"
    warn "created backend/.env from the example — its credentials are still blank"
  fi
  if grep -q '^DEFAULT_MODEL_VERSION=' "$ENV_FILE"; then
    # BSD and GNU sed disagree about -i, so write through a temp file.
    tmp="$(mktemp)"
    sed "s|^DEFAULT_MODEL_VERSION=.*|DEFAULT_MODEL_VERSION=$VERSION|" "$ENV_FILE" >"$tmp"
    mv "$tmp" "$ENV_FILE"
  else
    printf '\nDEFAULT_MODEL_VERSION=%s\n' "$VERSION" >>"$ENV_FILE"
  fi
  ok "backend/.env: DEFAULT_MODEL_VERSION=$VERSION"
  info "restart the API and the celery worker to load it"
  info "${DIM}in a deployed environment Firestore wins over this — activate there with:${NC}"
  info "${DIM}  curl -X POST -H \"Authorization: Bearer \$TOKEN\" \\${NC}"
  info "${DIM}    localhost:8000/api/v1/admin/models/$VERSION/activate${NC}"
else
  info "not serving yet — re-run with --activate, or POST /admin/models/$VERSION/activate"
fi
