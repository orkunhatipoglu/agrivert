#!/usr/bin/env bash
#
# Publish a trained model so everyone else can install it with
# scripts/model-download.sh. Run this on the machine that did the training.
#
#   ./scripts/model-upload.sh artifacts/vertical --version v3-vertical-20260809
#   ./scripts/model-upload.sh artifacts/vertical --version v3-... --pack-only
#
# Packs the artifacts into dist/, creates a GitHub Release with all three
# assets attached, and rewrites scripts/model-release.env so the download
# script and the READMEs point at the new bundle.

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ENV="$ROOT/scripts/model-release.env"
DIST="$ROOT/dist"

if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'
  BLD=$'\033[1m'; NC=$'\033[0m'
else
  RED=; GRN=; YLW=; BLU=; BLD=; NC=
fi
step() { printf '\n%s==>%s %s%s%s\n' "$BLU" "$NC" "$BLD" "$*" "$NC"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s✓%s %s\n' "$GRN" "$NC" "$*"; }
die()  { printf '    %s✗%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

ARTIFACTS=""; VERSION=""; NOTES=""
PACK_ONLY=0; ASSUME_YES=0

usage() {
  cat <<'EOF'
Pack and publish a trained model bundle.

USAGE
  ./scripts/model-upload.sh <artifacts-dir> --version NAME [options]

OPTIONS
  --version NAME   release tag and model version, e.g. v3-vertical-20260809
  --notes TEXT     release notes (default: generated from metadata.json)
  --pack-only      write dist/ and stop; do not touch GitHub
  --yes            do not ask before creating the release
  -h, --help       this text

Creating a release is public and awkward to undo, so it asks first.
--pack-only gives you the three files to upload by hand.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version)   VERSION="$2"; shift ;;
    --version=*) VERSION="${1#*=}" ;;
    --notes)     NOTES="$2"; shift ;;
    --notes=*)   NOTES="${1#*=}" ;;
    --pack-only) PACK_ONLY=1 ;;
    --yes|-y)    ASSUME_YES=1 ;;
    -h|--help)   usage; exit 0 ;;
    -*)          usage >&2; die "unknown option: $1" ;;
    *)           ARTIFACTS="$1" ;;
  esac
  shift
done

[ -n "$ARTIFACTS" ] || { usage >&2; die "no artifacts directory given"; }
[ -n "$VERSION" ]   || { usage >&2; die "--version is required"; }
[ -d "$ARTIFACTS" ] || die "$ARTIFACTS is not a directory"

PY="$ROOT/backend/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "no python found"

# --------------------------------------------------------------------- pack --

step "Packing $VERSION"
( cd "$ROOT" && "$PY" -m ml.bundle pack "$ARTIFACTS" --version "$VERSION" --out "$DIST" )

TARBALL="$DIST/$VERSION.tar.gz"
[ -f "$TARBALL" ] || die "expected $TARBALL, but pack did not produce it"
SHA="$(cut -d' ' -f1 <"$DIST/$VERSION.tar.gz.sha256")"
[ -n "$SHA" ] || die "could not read the hash from $DIST/$VERSION.tar.gz.sha256"

# metadata.json is the only honest source for what this model does; release
# notes written by hand have a habit of quoting the previous run's numbers.
SUMMARY="$("$PY" - "$ARTIFACTS/metadata.json" <<'PY'
import json, sys

m = json.load(open(sys.argv[1]))
mt, cal = m.get("metrics", {}), m.get("calibration", {})

acc = " / ".join(
    f"{d[len('test_'):]} {mt[d]['accuracy'] * 100:.1f}%"
    for d in ("test_studio", "test_field", "test_vertical")
    if d in mt
) or "no test metrics recorded"

verified = cal.get("verified") or {}
thr = f"threshold {cal.get('recommended_confidence_threshold')}"
if verified:
    thr += (
        f", answers {verified['coverage'] * 100:.1f}%"
        f" at {verified['selective_accuracy'] * 100:.1f}%"
    )
if cal.get("meets_target") is False:
    thr += " (MISSES its declared target -- do not advertise the target)"

print(f"{m.get('num_classes', '?')} classes")
print(acc)
print(thr)
PY
)"
CLASSES="$(printf '%s\n' "$SUMMARY" | sed -n 1p)"
ACC="$(printf '%s\n' "$SUMMARY" | sed -n 2p)"
THRESH="$(printf '%s\n' "$SUMMARY" | sed -n 3p)"

info "$CLASSES"
info "$ACC"
info "$THRESH"
info "sha256 $SHA"

if [ "$PACK_ONLY" -eq 1 ]; then
  ok "packed, nothing published"
  info "attach all three of $DIST/$VERSION.tar.gz, .tar.gz.sha256 and .manifest.json"
  exit 0
fi

# ------------------------------------------------------------------ release --

command -v gh >/dev/null 2>&1 || die "gh is not installed -- use --pack-only and upload by hand"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated -- run: gh auth login"

SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
[ -n "$NOTES" ] || NOTES="$CLASSES. $ACC. $THRESH."

step "Creating release $VERSION on $SLUG"
info "assets: $VERSION.tar.gz, .tar.gz.sha256, .manifest.json"
if [ "$ASSUME_YES" -eq 0 ]; then
  printf '    %spublish this release publicly? [y/N]%s ' "$YLW" "$NC"
  read -r reply
  case "$reply" in
    y|Y|yes) ;;
    *) die "aborted -- dist/ is still there if you want to upload by hand" ;;
  esac
fi

gh release create "$VERSION" "$DIST/$VERSION."* \
  --repo "$SLUG" --title "Agrivert model $VERSION" --notes "$NOTES"
ok "released"

# ------------------------------------------------------------------- repin ---

URL="https://github.com/$SLUG/releases/download/$VERSION/$VERSION.tar.gz"
tmp="$(mktemp)"
sed -e "s|^MODEL_VERSION=.*|MODEL_VERSION=\"$VERSION\"|" \
    -e "s|^MODEL_URL=.*|MODEL_URL=\"$URL\"|" \
    -e "s|^MODEL_SHA256=.*|MODEL_SHA256=\"$SHA\"|" "$RELEASE_ENV" >"$tmp"
mv "$tmp" "$RELEASE_ENV"
ok "repinned scripts/model-release.env to $VERSION"

step "Left to do by hand"
info "1. commit scripts/model-release.env"
info "2. update the release table in ml/README.md (metrics, threshold, sha256)"
info "3. tell everyone: ${BLD}./scripts/model-download.sh --activate${NC}"
