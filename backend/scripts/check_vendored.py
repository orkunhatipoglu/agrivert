#!/usr/bin/env python3
"""Check the vendored predict.py / data.py against the training repo's copies.

    python scripts/check_vendored.py /path/to/agrivert-ml

`backend/predict.py` and `backend/data.py` are copies of files owned by the
training project. Serving preprocessing must stay byte-identical to training
preprocessing (`build_eval_transform` especially) or the model sees inputs it
was never trained on — a silent accuracy loss with no error anywhere.

Exit codes: 0 identical, 1 drift found, 2 could not compare.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

VENDORED = ("predict.py", "data.py")

BACKEND_DIR = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "source",
        type=Path,
        help="Path to the agrivert-ml checkout holding the authoritative copies",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="Copy the source files over the vendored ones instead of just reporting",
    )
    args = ap.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        print(f"error: {source} is not a directory", file=sys.stderr)
        return 2

    drift = []
    for name in VENDORED:
        src = source / name
        dst = BACKEND_DIR / name
        if not src.is_file():
            print(f"error: {src} not found", file=sys.stderr)
            return 2
        if not dst.is_file():
            print(f"error: vendored {dst} missing", file=sys.stderr)
            return 2

        if digest(src) == digest(dst):
            print(f"  ok        {name}")
            continue

        drift.append(name)
        if args.update:
            dst.write_bytes(src.read_bytes())
            print(f"  UPDATED   {name}")
        else:
            print(f"  DRIFTED   {name}")

    if not drift:
        print("\nvendored copies match the training repo.")
        return 0

    if args.update:
        print(f"\nupdated {len(drift)} file(s). Re-run the backend tests.")
        return 0

    print(
        f"\n{len(drift)} file(s) differ from {source}.\n"
        "Serving preprocessing may no longer match training. Re-run with "
        "--update to sync, then re-run the tests.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
