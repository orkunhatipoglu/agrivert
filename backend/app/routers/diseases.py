"""Disease knowledge base (ROUTES.md flaw #5).

STUB STATUS: reads are implemented against Firestore, but the collection ships
EMPTY of agronomic content. scripts/seed_diseases.py creates one document per
model class with blank description/symptoms/treatment fields for a human to
fill in.

That emptiness is deliberate. This endpoint is what turns a label into advice
a farmer acts on in a real field; generating that text without a cited
agronomic source would produce confident, plausible, unreviewed guidance —
the same failure mode project_context.md §2.7 rejects for predictions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import CurrentUser, get_current_user
from app.firebase import COLLECTION_DISEASES, get_db
from app.schemas.diseases import Disease, DiseaseList, DiseaseSummary

router = APIRouter(prefix="/diseases", tags=["diseases"])


@router.get(
    "",
    response_model=DiseaseList,
    summary="List all diseases the model can detect",
)
async def list_diseases(_: CurrentUser = Depends(get_current_user)) -> DiseaseList:
    docs = get_db().collection(COLLECTION_DISEASES).stream()
    items = [
        DiseaseSummary(
            disease_id=d["disease_id"],
            raw_label=d["raw_label"],
            crop=d["crop"],
            condition=d["condition"],
            healthy=d.get("healthy", False),
            field_validated=d.get("field_validated", False),
        )
        for d in (snap.to_dict() for snap in docs)
    ]
    items.sort(key=lambda i: (i.crop, i.condition))
    return DiseaseList(items=items)


@router.get("/{disease_id}", response_model=Disease, summary="Disease details")
async def get_disease(
    disease_id: str, _: CurrentUser = Depends(get_current_user)
) -> Disease:
    snap = get_db().collection(COLLECTION_DISEASES).document(disease_id).get()
    if not snap.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"disease {disease_id!r} not found; run scripts/seed_diseases.py "
                "to populate the knowledge base from the active model's labels"
            ),
        )
    return Disease(**snap.to_dict())
