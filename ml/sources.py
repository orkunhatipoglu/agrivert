"""Dataset adapters for the vertical-agriculture blend.

Each adapter turns one source's on-disk layout into a uniform
`list[Sample]`, mapping that source's label spelling onto the canonical
taxonomy. All the per-dataset weirdness is quarantined here.

The three domains matter more than the five sources:

    studio    PlantVillage — single leaf, plain background, even light.
              Plentiful and nearly useless on its own.
    field     PlantDoc / PlantWild / PlantSeg — outdoor, cluttered, in the
              wild. Closer to reality but not vertical farming.
    vertical  Hydroponic and greenhouse lettuce — the actual deployment
              domain: artificial light, dense racks, close-range camera.
              By far the scarcest, and the one worth optimising for.

Checkpoint selection targets `vertical` for exactly the reason the previous
pipeline targeted `field`: the domain you validate on is the domain you get.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ml.taxonomy import (
    LETTUCE_GREENHOUSE_MAP,
    LETTUCE_HYDRO_MAP,
    PLANTDOC_MAP,
    PLANTVILLAGE_MAP,
    PLANTWILD_MAP,
)

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DOMAIN_STUDIO = "studio"
DOMAIN_FIELD = "field"
DOMAIN_VERTICAL = "vertical"


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str  # canonical class
    domain: str
    source: str


def _images_in(directory: Path):
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _find_dir(root: Path, *names: str) -> Path | None:
    """Locate a subdirectory case-insensitively, at any depth.

    Kaggle/HF archives vary in how deeply they nest and in casing, so the
    adapters look for a landmark folder rather than assuming a fixed path.
    """
    wanted = {n.lower() for n in names}
    if root.name.lower() in wanted:
        return root
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in wanted:
            return p
    return None


def _by_class_folder(
    root: Path, mapping: dict[str, str], domain: str, source: str
) -> list[Sample]:
    """Standard `<root>/<class-name>/*.jpg` layout."""
    samples: list[Sample] = []
    unmapped: dict[str, int] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        canonical = mapping.get(child.name)
        if canonical is None:
            n = sum(1 for _ in _images_in(child))
            if n:
                unmapped[child.name] = n
            continue
        for image in _images_in(child):
            samples.append(Sample(image, canonical, domain, source))
    if unmapped:
        log.debug(
            "%s: skipped %d out-of-scope class folder(s)", source, len(unmapped)
        )
    return samples


def load_plantvillage(root: Path) -> list[Sample]:
    """PlantVillage: `color/<Crop>___<Condition>/`. Studio domain."""
    base = _find_dir(root, "color") or root
    return _by_class_folder(base, PLANTVILLAGE_MAP, DOMAIN_STUDIO, "plantvillage")


def load_plantdoc(root: Path) -> list[Sample]:
    """PlantDoc: `train/` and `test/` with free-form class names. Field."""
    out: list[Sample] = []
    for split in ("train", "test"):
        d = _find_dir(root, split)
        if d:
            out += _by_class_folder(d, PLANTDOC_MAP, DOMAIN_FIELD, "plantdoc")
    return out


def load_combined_pv_pd(root: Path) -> list[Sample]:
    """srabon00 combined PV+PD.

    Only its `train/` split is usable: `test/` is a flat folder of Roboflow
    exports with no class subdirectories, so the labels aren't recoverable
    from the layout. Its train classes use PlantVillage spelling.
    """
    train = _find_dir(root, "train")
    if train is None:
        log.warning("combined dataset: no train/ directory under %s", root)
        return []
    samples = _by_class_folder(train, PLANTVILLAGE_MAP, DOMAIN_STUDIO, "combined")
    # PlantDoc-spelled folders may sit alongside after the merge.
    samples += _by_class_folder(train, PLANTDOC_MAP, DOMAIN_FIELD, "combined")
    return samples


def load_plantwild(root: Path) -> list[Sample]:
    """PlantWild v2: `plantwild_v2/<plant> <disease>/`. Field domain.

    Disease-only — this source has no healthy class for any crop.
    """
    base = _find_dir(root, "plantwild_v2", "plantwild") or root
    return _by_class_folder(base, PLANTWILD_MAP, DOMAIN_FIELD, "plantwild")


def load_plantseg(root: Path, use_masks: bool = False) -> list[Sample]:
    """PlantSeg: images named `<plant>_<disease>_<n>.jpg` plus masks.

    The class is encoded in the filename, so labels come from a regex against
    the taxonomy keys rather than from the directory. Segmentation masks are
    ignored for classification (`use_masks` is reserved for a future
    crop-to-lesion variant, which would fight background reliance harder).
    """
    images_dir = _find_dir(root, "images")
    if images_dir is None:
        log.warning("plantseg: no images/ directory under %s", root)
        return []

    # Longest key first so "tomato leaf mold" wins over a shorter prefix.
    keys = sorted(PLANTWILD_MAP, key=len, reverse=True)
    patterns = [(re.sub(r"\s+", "_", k), PLANTWILD_MAP[k]) for k in keys]

    samples: list[Sample] = []
    for image in _images_in(images_dir):
        stem = image.stem.lower()
        for prefix, canonical in patterns:
            if stem.startswith(prefix):
                samples.append(Sample(image, canonical, DOMAIN_FIELD, "plantseg"))
                break
    return samples


def load_lettuce_hydroponic(root: Path) -> list[Sample]:
    """Locarno NFT hydroponic lettuce: `Lettuce disease/<Class>/`.

    The only source shot inside an actual vertical system, and the only
    source of nutrient-deficiency classes. Tiny (~209 images) but it defines
    the deployment domain, so it is oversampled hard during training.
    """
    base = _find_dir(root, "lettuce disease") or root
    return _by_class_folder(
        base, LETTUCE_HYDRO_MAP, DOMAIN_VERTICAL, "lettuce_hydroponic"
    )


def load_lettuce_greenhouse(root: Path) -> list[Sample]:
    """wingsdong greenhouse lettuce: `day*/<time>/picture/{healthy,bad,all}/`.

    Three traps, all handled here:
      * 18.6GB of the download is .mp4 — ignored entirely.
      * `all/` duplicates the images in `healthy/` and `bad/`; including it
        would double-count and leak between splits.
      * `bad/` carries no specific disease, so it cannot be mapped to a
        disease class without inventing a label. Only `healthy/` is used.
    """
    samples: list[Sample] = []
    seen: set[str] = set()
    for picture_dir in root.rglob("picture"):
        if not picture_dir.is_dir():
            continue
        for label_dir in picture_dir.iterdir():
            if not label_dir.is_dir():
                continue
            canonical = LETTUCE_GREENHOUSE_MAP.get(label_dir.name.lower())
            if canonical is None:
                continue  # skips `all/` and `bad/`
            for image in _images_in(label_dir):
                # Same frame reappears across day folders; dedupe by name+size.
                key = f"{image.name}:{image.stat().st_size}"
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    Sample(image, canonical, DOMAIN_VERTICAL, "lettuce_greenhouse")
                )
    return samples


LOADERS = {
    "plantvillage": load_plantvillage,
    "plantdoc": load_plantdoc,
    "combined": load_combined_pv_pd,
    "plantwild": load_plantwild,
    "plantseg": load_plantseg,
    "lettuce_hydroponic": load_lettuce_hydroponic,
    "lettuce_greenhouse": load_lettuce_greenhouse,
}

# kagglehub slugs / HF repo, so the training script can fetch what's missing.
SOURCE_REFS = {
    "plantvillage": ("kaggle", "abdallahalidev/plantvillage-dataset"),
    "plantdoc": ("kaggle", "manojkumarcs28/plantdoc-dataset"),
    "combined": ("kaggle", "srabon00/combine-plant-disease-dataset-pv-pd-fp"),
    "plantseg": ("kaggle", "weitianqi/plantseg"),
    "lettuce_hydroponic": ("kaggle", "rathorhome/lettuce-disease"),
    "lettuce_greenhouse": ("kaggle", "wingsdong/lettuce-diseases-and-pests"),
    "plantwild": ("hf", "uqtwei2/PlantWild"),
}

# Sources whose download cost is wildly out of proportion to what they yield.
EXPENSIVE_SOURCES = {
    "lettuce_greenhouse": (
        "18.6 GB download, of which ~18.6 GB is video; yields only a few "
        "hundred usable healthy-lettuce stills"
    ),
    "combined": (
        "7.9 GB, and its train split is PlantVillage+PlantDoc which you can "
        "get directly; its test split has no class folders"
    ),
}


def collect(roots: dict[str, Path]) -> list[Sample]:
    """Run every adapter for which a root was provided."""
    samples: list[Sample] = []
    for name, root in roots.items():
        loader = LOADERS.get(name)
        if loader is None:
            raise KeyError(f"unknown source {name!r}; known: {sorted(LOADERS)}")
        found = loader(Path(root))
        log.info("%-20s %6d images from %s", name, len(found), root)
        samples += found
    return samples
