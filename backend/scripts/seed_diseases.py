#!/usr/bin/env python3
"""Seed the disease knowledge base from the active model's labels.json.

    python scripts/seed_diseases.py --version v1-blended-20260808
    python scripts/seed_diseases.py --dry-run

Creates one Firestore document per class the model can predict, keyed by
raw_label, carrying the structural facts we actually know: crop, condition,
healthy flag, and whether the class has real field training data
(project_context.md §2.3 — 10 of the 38 classes are studio-only).

The agronomic content fields (description, symptoms, treatment, prevention,
severity) are created EMPTY and left for a human.

Why empty: this text is what a farmer reads before spraying something on a
real crop. Auto-generating it would produce fluent, plausible, uncited advice
with no way for the reader to tell it apart from reviewed guidance. The
`content_reviewed` flag exists so the frontend can refuse to render unreviewed
entries as advice.

Re-running is safe: existing documents keep their content, and only the
structural fields are refreshed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml import registry  # noqa: E402

CONTENT_FIELDS = {
    "description": None,
    "symptoms": [],
    "treatment": [],
    "prevention": [],
    "severity": "unknown",
    "references": [],
    "content_reviewed": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None, help="Model version (default: active)")
    ap.add_argument("--dry-run", action="store_true", help="Print, don't write")
    args = ap.parse_args()

    try:
        mv = registry.get_version(args.version) if args.version else registry.resolve_active_version()
    except Exception as exc:
        print(f"error: could not resolve model version: {exc}", file=sys.stderr)
        return 1

    labels = json.loads((mv.path / "labels.json").read_text())
    metadata = mv.load_metadata()
    field_classes = set(metadata.get("field_covered_classes", []))

    print(f"seeding from model version {mv.version} ({len(labels)} classes)")

    docs = []
    for _idx, meta in sorted(labels.items(), key=lambda kv: int(kv[0])):
        raw_label = meta["raw_label"]
        docs.append(
            {
                "disease_id": raw_label,
                "raw_label": raw_label,
                "crop": meta["crop"],
                "condition": meta["condition"],
                "healthy": meta["healthy"],
                "field_validated": raw_label in field_classes,
                **CONTENT_FIELDS,
            }
        )

    studio_only = [d for d in docs if not d["field_validated"]]

    if args.dry_run:
        for d in docs:
            flag = "" if d["field_validated"] else "  [studio-only]"
            print(f"  {d['crop']:<28} {d['condition']}{flag}")
        print(f"\n{len(docs)} documents would be written ({len(studio_only)} studio-only)")
        return 0

    from app.firebase import COLLECTION_DISEASES, get_db

    db = get_db()
    collection = db.collection(COLLECTION_DISEASES)

    written = preserved = 0
    batch = db.batch()
    for i, doc in enumerate(docs):
        ref = collection.document(doc["disease_id"])
        existing = ref.get()
        if existing.exists:
            # Never clobber human-written content on a re-seed.
            current = existing.to_dict()
            for key in CONTENT_FIELDS:
                if key in current:
                    doc[key] = current[key]
            preserved += 1
        batch.set(ref, doc, merge=True)
        written += 1
        if (i + 1) % 400 == 0:  # Firestore batch cap is 500
            batch.commit()
            batch = db.batch()
    batch.commit()

    print(f"\nwrote {written} disease documents ({preserved} already existed)")
    print(f"{len(studio_only)} are studio-only (no field training data)")
    print(
        "\nAll content fields are empty and content_reviewed=false. "
        "Fill them in before the frontend renders /diseases as advice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
