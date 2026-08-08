#!/usr/bin/env python3
"""Register a trained model version with the backend.

This is the whole model-swap workflow. After a retrain:

    python scripts/register_model.py ../artifacts --version v2-blended-20260901
    python scripts/register_model.py ../artifacts --version v2-... --activate

It copies the three files the serving path needs into
backend/models/<version>/, records the version (with its metrics) in
Firestore, and optionally makes it active. No backend code changes.

Note the ONNX export is NOT copied: the current serving path loads best.pt
through predict.py. Keep the ONNX file with your training artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Make `app` importable when run from backend/ or backend/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ml.registry import REQUIRED_FILES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", type=Path, help="Source artifacts dir")
    ap.add_argument("--version", required=True, help="Version name, e.g. v1-blended-20260808")
    ap.add_argument("--activate", action="store_true", help="Make this the serving version")
    ap.add_argument("--notes", default=None)
    ap.add_argument(
        "--skip-firestore",
        action="store_true",
        help="Copy files only; useful for local dev with DEFAULT_MODEL_VERSION",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite an existing version dir")
    args = ap.parse_args()

    src = args.artifacts.resolve()
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1

    missing = [f for f in REQUIRED_FILES if not (src / f).is_file()]
    if missing:
        print(
            f"error: {src} is missing required file(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    if "/" in args.version or "\\" in args.version or args.version.startswith("."):
        print(f"error: invalid version name {args.version!r}", file=sys.stderr)
        return 1

    dest = Path(get_settings().model_registry_dir) / args.version
    if dest.exists() and not args.force:
        print(
            f"error: {dest} already exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(src / name, dest / name)
        print(f"  copied {name}")

    metadata = json.loads((dest / "metadata.json").read_text())
    metrics = metadata.get("metrics", {})
    print(f"\nregistered {args.version} at {dest}")

    field = metrics.get("test_field", {})
    studio = metrics.get("test_studio", {})
    if field:
        print(
            f"  test_field : acc={field.get('accuracy'):.4f} "
            f"macro_f1={field.get('macro_f1'):.4f}   <-- judge on this one"
        )
    if studio:
        print(
            f"  test_studio: acc={studio.get('accuracy'):.4f} "
            f"macro_f1={studio.get('macro_f1'):.4f}   (inflated; see caveat)"
        )

    if args.skip_firestore:
        print(
            f"\nskipped Firestore. For local dev set: "
            f"DEFAULT_MODEL_VERSION={args.version}"
        )
        return 0

    try:
        from app.repositories import models as model_repo

        model_repo.register(args.version, metrics=metrics, notes=args.notes)
        print("  recorded in Firestore")
        if args.activate:
            model_repo.activate(args.version)
            print(f"  ACTIVATED {args.version}")
            print(
                "\nrestart the Celery workers so they pick up the new version "
                "(each worker process caches its loaded model)."
            )
    except Exception as exc:
        print(f"\nwarning: files copied but Firestore update failed: {exc}", file=sys.stderr)
        print(f"set DEFAULT_MODEL_VERSION={args.version} to serve it anyway.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
