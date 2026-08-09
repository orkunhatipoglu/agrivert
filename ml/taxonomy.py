"""Vertical-agriculture crop taxonomy.

Scope is deliberately narrow: only crops actually grown in vertical farms.
The 38-class PlantVillage model this replaces spent most of its capacity on
apple, corn, grape, orange and soybean — none of which go in a grow rack.

WHAT THE SOURCE DATASETS ACTUALLY CONTAIN (verified against the real class
lists, not assumed):

  Covered      lettuce, basil, strawberry, tomato, bell pepper, celery,
               cabbage
  NOT covered  kale, spinach, arugula, mint, cilantro, thyme, microgreens

The second row matters. Those crops appear in NONE of the five source
datasets, so no class can be built for them and the model will silently
mis-assign their photos to whatever looks closest. `UNCOVERED_TARGETS` keeps
that fact in code rather than in someone's memory; `verify_coverage()` prints
it at training time.
"""

from __future__ import annotations

from ml.contract import LABEL_SEPARATOR

# --- Crop groups (the user-facing categories) ------------------------------

CROP_GROUPS: dict[str, tuple[str, ...]] = {
    "leafy_greens": ("Lettuce", "Cabbage", "Celery"),
    "herbs": ("Basil",),
    "fruiting": ("Tomato", "Strawberry", "Pepper_bell"),
}

VERTICAL_CROPS: tuple[str, ...] = tuple(
    crop for crops in CROP_GROUPS.values() for crop in crops
)

# Requested for the product but absent from every source dataset. Kept
# explicit so nobody assumes silence means coverage.
UNCOVERED_TARGETS: tuple[str, ...] = (
    "Kale",
    "Spinach",
    "Arugula",
    "Mint",
    "Cilantro",
    "Thyme",
    "Microgreens",
)

# --- The class vocabulary --------------------------------------------------
# Every class the model can emit. Healthy classes are listed only where a
# source actually provides healthy images — inventing a healthy class with no
# data would make the model claim health it has never seen.

VERTICAL_CLASSES: tuple[str, ...] = (
    # Lettuce — diseases from PlantWild/PlantSeg, deficiencies + healthy from
    # the hydroponics dataset (the only true vertical-farm source we have).
    "Lettuce___healthy",
    "Lettuce___Downy_mildew",
    "Lettuce___Mosaic_virus",
    "Lettuce___Nitrogen_deficiency",
    "Lettuce___Potassium_deficiency",
    # Sclerotinia (lettuce drop) — from the Roboflow greenhouse set, the only
    # source that has it. Kept next to the other lettuce classes rather than
    # appended at the end: checkpoints are warm-started by class *name*
    # (see load_checkpoint_into), so this tuple's order is free to stay
    # readable and does not have to preserve historical indices.
    "Lettuce___Sclerotinia_rot",
    "Lettuce___Wilt",
    # Basil — disease only; no healthy basil exists in any source.
    "Basil___Downy_mildew",
    # Cabbage (brassica stand-in for the kale/leafy-brassica group).
    "Cabbage___Alternaria_leaf_spot",
    "Cabbage___Black_rot",
    "Cabbage___Downy_mildew",
    # Celery
    "Celery___Anthracnose",
    "Celery___Early_blight",
    # Tomato — the best-covered crop by a wide margin.
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Mosaic_virus",
    "Tomato___Yellow_Leaf_Curl_Virus",
    # Strawberry
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___Anthracnose",
    # Bell pepper
    "Pepper_bell___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___Blossom_end_rot",
    "Pepper_bell___Frogeye_leaf_spot",
    "Pepper_bell___Powdery_mildew",
)

# --- Source label -> canonical class ---------------------------------------
# Each source spells its labels differently. These maps are the whole reason
# the pipeline can blend them; they are data, not logic, so they are readable
# and reviewable at a glance.

# PlantVillage / PlantDoc already use the ___ convention.
PLANTVILLAGE_MAP: dict[str, str] = {
    "Tomato___healthy": "Tomato___healthy",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato___Mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Yellow_Leaf_Curl_Virus",
    "Strawberry___healthy": "Strawberry___healthy",
    "Strawberry___Leaf_scorch": "Strawberry___Leaf_scorch",
    "Pepper,_bell___healthy": "Pepper_bell___healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper_bell___Bacterial_spot",
}

# PlantDoc's free-form folder names.
PLANTDOC_MAP: dict[str, str] = {
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites",
    "Strawberry leaf": "Strawberry___healthy",
    "Bell_pepper leaf": "Pepper_bell___healthy",
    "Bell_pepper leaf spot": "Pepper_bell___Bacterial_spot",
}

# PlantWild v2 folders and PlantSeg's `Disease` column share one vocabulary
# (same authors), so one map serves both.
PLANTWILD_MAP: dict[str, str] = {
    "lettuce downy mildew": "Lettuce___Downy_mildew",
    "lettuce mosaic virus": "Lettuce___Mosaic_virus",
    "basil downy mildew": "Basil___Downy_mildew",
    "cabbage alternaria leaf spot": "Cabbage___Alternaria_leaf_spot",
    "cabbage black rot": "Cabbage___Black_rot",
    "cabbage downy mildew": "Cabbage___Downy_mildew",
    "celery anthracnose": "Celery___Anthracnose",
    "celery early blight": "Celery___Early_blight",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Yellow_Leaf_Curl_Virus",
    "strawberry anthracnose": "Strawberry___Anthracnose",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "bell pepper bacterial spot": "Pepper_bell___Bacterial_spot",
    "bell pepper blossom end rot": "Pepper_bell___Blossom_end_rot",
    "bell pepper frogeye leaf spot": "Pepper_bell___Frogeye_leaf_spot",
    "bell pepper powdery mildew": "Pepper_bell___Powdery_mildew",
}

# Locarno hydroponic lettuce (rathorhome). The only source photographed in an
# actual vertical NFT system, and the only source of nutrient-deficiency
# classes.
LETTUCE_HYDRO_MAP: dict[str, str] = {
    # The folder names as they actually ship in kagglehub version 2. These are
    # the ones that matter; the spellings below are aliases kept for other
    # redistributions. Getting these wrong is expensive and silent: an
    # unmapped folder is skipped, and because this is the ONLY true vertical
    # source, mismatched keys once dropped 147 of 209 images and three whole
    # classes while the run still looked healthy.
    "Healthy": "Lettuce___healthy",
    "N Deficient": "Lettuce___Nitrogen_deficiency",
    "K Deficient": "Lettuce___Potassium_deficiency",
    "Wilt fungal": "Lettuce___Wilt",
    # Alternate spellings seen in other redistributions of this dataset.
    "N deficiency": "Lettuce___Nitrogen_deficiency",
    "K deficiency": "Lettuce___Potassium_deficiency",
    "Wilt": "Lettuce___Wilt",
    # Seen as bare initials in some redistributions of this dataset.
    "H": "Lettuce___healthy",
    "N": "Lettuce___Nitrogen_deficiency",
    "K": "Lettuce___Potassium_deficiency",
    "W": "Lettuce___Wilt",
}

# wingsdong greenhouse lettuce: only healthy vs. not. `bad` carries no
# specific disease, so it cannot be mapped to a disease class — only the
# healthy half is usable without inventing a label.
LETTUCE_GREENHOUSE_MAP: dict[str, str] = {
    "healthy": "Lettuce___healthy",
}

# Roboflow `phs/lettuce_disease` (MIT), exported as multiclass. The export is
# one-hot over five columns, but only three of them are health states:
# `growing` and `raising_seeding` are GROWTH STAGES that co-occur with
# `health`, not conditions of their own. Mapping them as labels would teach
# the model to answer "seedling" when asked what is wrong with a plant.
#
# The combinations are clean and mutually exclusive in practice — no image
# carries both diseases, and no diseased image is also marked healthy:
#   growing+health            5803  -> healthy
#   health+raising_seeding    1914  -> healthy
#   sclerotinia_rot           1184  -> Sclerotinia
#   downy_mildew              1078  -> Downy mildew
LETTUCE_ROBOFLOW_MAP: dict[str, str] = {
    "sclerotinia_rot": "Lettuce___Sclerotinia_rot",
    "downy_mildew": "Lettuce___Downy_mildew",
    "health": "Lettuce___healthy",
}

# Growth stages, deliberately not labels. Listed so the adapter can tell
# "column I chose to ignore" apart from "column I have never seen".
LETTUCE_ROBOFLOW_STAGES: frozenset[str] = frozenset({"growing", "raising_seeding"})

# ashishjstar/lettuce-diseases. Read the counts before trusting this source:
# of ~2337 images, 1123 are Healthy and 1106 are a *weed* (Shepherd's purse,
# not a lettuce condition at all). The five disease folders hold 6-30 images
# each, which is too few to learn and more than enough to produce confident
# nonsense, so only the classes that already exist elsewhere are mapped.
#
# Deliberately unmapped: Bacterial (20), Powdery_mildew (18),
# Septoria_blight (19) — no corresponding class exists and 20 images cannot
# create one; Shepherd_purse_weeds (1106) — a weed, and the images are 119x119
# thumbnails.
LETTUCE_KAGGLE_MAP: dict[str, str] = {
    "Healthy": "Lettuce___healthy",
    "Downy_mildew_on_lettuce": "Lettuce___Downy_mildew",
    "Wilt_and_leaf_blight_on_lettuce": "Lettuce___Wilt",
}

CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(VERTICAL_CLASSES)}


def crop_of(canonical_class: str) -> str:
    return canonical_class.partition(LABEL_SEPARATOR)[0]


def group_of(canonical_class: str) -> str | None:
    """Which CROP_GROUPS bucket a class belongs to."""
    crop = crop_of(canonical_class)
    for group, crops in CROP_GROUPS.items():
        if crop in crops:
            return group
    return None


def validate_taxonomy() -> None:
    """Fail loudly if a map points at a class that doesn't exist.

    A typo here would otherwise drop every image of that class on the floor
    and train a quietly worse model.
    """
    known = set(VERTICAL_CLASSES)
    for name, mapping in (
        ("PLANTVILLAGE_MAP", PLANTVILLAGE_MAP),
        ("PLANTDOC_MAP", PLANTDOC_MAP),
        ("PLANTWILD_MAP", PLANTWILD_MAP),
        ("LETTUCE_HYDRO_MAP", LETTUCE_HYDRO_MAP),
        ("LETTUCE_GREENHOUSE_MAP", LETTUCE_GREENHOUSE_MAP),
        ("LETTUCE_ROBOFLOW_MAP", LETTUCE_ROBOFLOW_MAP),
        ("LETTUCE_KAGGLE_MAP", LETTUCE_KAGGLE_MAP),
    ):
        for source_label, canonical in mapping.items():
            if canonical not in known:
                raise ValueError(
                    f"{name}[{source_label!r}] -> {canonical!r}, which is not "
                    "in VERTICAL_CLASSES"
                )

    unreachable = known - {
        c
        for mapping in (
            PLANTVILLAGE_MAP,
            PLANTDOC_MAP,
            PLANTWILD_MAP,
            LETTUCE_HYDRO_MAP,
            LETTUCE_GREENHOUSE_MAP,
            LETTUCE_ROBOFLOW_MAP,
            LETTUCE_KAGGLE_MAP,
        )
        for c in mapping.values()
    }
    if unreachable:
        raise ValueError(
            "these classes are declared but no source maps to them, so they "
            f"can never receive an image: {sorted(unreachable)}"
        )


def coverage_report() -> str:
    """Human-readable summary of what this taxonomy does and does not cover."""
    lines = [
        f"vertical-ag taxonomy: {len(VERTICAL_CLASSES)} classes across "
        f"{len(VERTICAL_CROPS)} crops",
    ]
    for group, crops in CROP_GROUPS.items():
        n = sum(1 for c in VERTICAL_CLASSES if crop_of(c) in crops)
        lines.append(f"  {group:14} {', '.join(crops):40} {n} classes")
    lines.append("")
    lines.append(
        "NOT COVERED (requested, but absent from all source datasets): "
        + ", ".join(UNCOVERED_TARGETS)
    )
    lines.append(
        "  Photos of these crops will be forced into the nearest available "
        "class. Collect data before promising support."
    )
    healthy = [c for c in VERTICAL_CLASSES if c.endswith("healthy")]
    no_healthy = sorted(
        {crop_of(c) for c in VERTICAL_CLASSES}
        - {crop_of(c) for c in healthy}
    )
    if no_healthy:
        lines.append("")
        lines.append(
            "Crops with NO healthy class (disease-only, so the model can "
            "never call them well): " + ", ".join(no_healthy)
        )
    return "\n".join(lines)
