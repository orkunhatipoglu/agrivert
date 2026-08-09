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

import csv
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ml.taxonomy import (
    LETTUCE_GREENHOUSE_MAP,
    LETTUCE_HYDRO_MAP,
    LETTUCE_KAGGLE_MAP,
    LETTUCE_ROBOFLOW_MAP,
    LETTUCE_ROBOFLOW_STAGES,
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

    `names` is a PRIORITY ORDER, not a set: the first name that matches
    anywhere wins. PlantWild depends on this. It ships a v1 `plantwild/` flat
    dump alongside the usable `plantwild_v2/`, and the directory it extracts
    into is itself named `plantwild` — so treating the names as an unordered
    set matched the wrapper directory and returned a level of the tree with
    no class folders in it, yielding zero images without an error.
    """
    for name in names:
        wanted = name.lower()
        if root.name.lower() == wanted:
            return root
        matches = [
            p for p in root.rglob("*") if p.is_dir() and p.name.lower() == wanted
        ]
        if matches:
            # Shallowest wins, ties broken lexicographically, so a nested
            # duplicate cannot make the choice vary between runs.
            return min(matches, key=lambda p: (len(p.parts), str(p)))
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


def load_lettuce_roboflow(root: Path) -> list[Sample]:
    """Roboflow `phs/lettuce_disease`, multiclass export.

    Layout is `<split>/_classes.csv` plus the images beside it. The CSV is
    one-hot over five columns, two of which (`growing`, `raising_seeding`) are
    growth stages rather than health states — see LETTUCE_ROBOFLOW_MAP.

    Disease wins over `health` when both are set. That never happens in this
    export, but the rule is written down rather than assumed: a future
    re-export that does mark a diseased plant as partially healthy must not
    silently start labelling diseased lettuce as healthy.

    Roboflow's own train/valid/test split is ignored on purpose. Splitting is
    this project's job (group-aware, see ml.dedup) and mixing two split
    schemes is how images end up on both sides.
    """
    samples: list[Sample] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_path = split_dir / "_classes.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                filename = row.get("filename")
                if not filename:
                    continue
                # Roboflow pads the header with spaces: " health", " growing".
                on = {k.strip() for k, v in row.items()
                      if k != "filename" and (v or "").strip() == "1"}
                labels = [LETTUCE_ROBOFLOW_MAP[c] for c in on
                          if c in LETTUCE_ROBOFLOW_MAP]
                diseases = [c for c in labels if not c.endswith("healthy")]
                canonical = diseases[0] if diseases else (labels[0] if labels else None)
                unknown = on - set(LETTUCE_ROBOFLOW_MAP) - LETTUCE_ROBOFLOW_STAGES
                if unknown:
                    log.warning(
                        "lettuce_roboflow: unrecognised column(s) %s in %s — "
                        "the export schema changed; check the mapping",
                        sorted(unknown), csv_path,
                    )
                if canonical is None:
                    continue
                image = split_dir / filename
                if image.is_file():
                    samples.append(
                        Sample(image, canonical, DOMAIN_VERTICAL, "lettuce_roboflow")
                    )
    return samples


def load_lettuce_kaggle(root: Path) -> list[Sample]:
    """ashishjstar/lettuce-diseases: `Lettuce_disease_datasets/<Class>/`.

    Most of this dataset is unusable — see LETTUCE_KAGGLE_MAP. Only the three
    folders that map onto classes we already have are loaded; the rest are
    6-30 images each, or a weed.
    """
    base = _find_dir(root, "Lettuce_disease_datasets") or root
    return _by_class_folder(
        base, LETTUCE_KAGGLE_MAP, DOMAIN_VERTICAL, "lettuce_kaggle"
    )


LOADERS = {
    "plantvillage": load_plantvillage,
    "plantdoc": load_plantdoc,
    "combined": load_combined_pv_pd,
    "plantwild": load_plantwild,
    "plantseg": load_plantseg,
    "lettuce_hydroponic": load_lettuce_hydroponic,
    "lettuce_greenhouse": load_lettuce_greenhouse,
    "lettuce_roboflow": load_lettuce_roboflow,
    "lettuce_kaggle": load_lettuce_kaggle,
}

# kagglehub slugs / HF repo, so the training script can fetch what's missing.
SOURCE_REFS = {
    "plantvillage": ("kaggle", "abdallahalidev/plantvillage-dataset"),
    "plantdoc": ("kaggle", "manojkumarcs28/plantdoc-dataset"),
    "combined": ("kaggle", "srabon00/combine-plant-disease-dataset-pv-pd-fp"),
    "plantseg": ("kaggle", "weitianqi/plantseg"),
    "lettuce_hydroponic": ("kaggle", "rathorhome/lettuce-disease"),
    "lettuce_greenhouse": ("kaggle", "wingsdong/lettuce-diseases-and-pests"),
    "lettuce_kaggle": ("kaggle", "ashishjstar/lettuce-diseases"),
    "plantwild": ("hf", "uqtwei2/PlantWild"),
    # lettuce_roboflow has no auto-download: it comes from a Roboflow version
    # export, which has to be generated in a workspace before it exists. Pass
    # --root lettuce_roboflow=<dir> pointing at the unzipped export. See
    # ml/README.md for how to regenerate it.
}

# Sources that arrive as archives rather than as a directory tree. HuggingFace
# `snapshot_download` hands back the repo's *files*, so PlantWild lands as two
# zips and every adapter that expects folders silently finds nothing.
#
# The value picks which archive to use when a repo ships several. PlantWild
# ships both: `plantwild.zip` is the v1 flat `images/` dump with no class
# folders (unusable — the labels live in a separate annotation file), while
# `plantwild_v2.zip` has the `plantwild_v2/<plant> <disease>/` layout
# `load_plantwild` reads. Extracting both would also leave `_find_dir`
# choosing between two landmark directories.
SOURCE_ARCHIVES = {
    "plantwild": "plantwild_v2.zip",
}


def ensure_extracted(name: str, root: Path, cache: Path) -> Path:
    """Unpack an archive-shaped source, returning the directory to scan.

    A no-op for sources that already arrive as directories, so it is safe to
    call for every source. Extraction goes to `cache/extracted/<name>` rather
    than into the download directory, because the HuggingFace cache is
    content-addressed and writing into a snapshot corrupts its bookkeeping.
    """
    root = Path(root)
    archives = sorted(root.glob("*.zip"))
    if not archives:
        return root

    preferred = SOURCE_ARCHIVES.get(name)
    if preferred:
        archives = [a for a in archives if a.name == preferred]
        if not archives:
            raise FileNotFoundError(
                f"{name}: expected {preferred} in {root}; found "
                f"{[p.name for p in sorted(root.glob('*.zip'))]}"
            )

    dest = Path(cache) / "extracted" / name
    # A marker file, not just dest.exists(): an extraction interrupted halfway
    # leaves a directory that looks complete and would train on partial data.
    marker = dest / ".extraction-complete"
    if marker.exists():
        log.info("%-20s already extracted at %s", name, dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        log.info("extracting %s -> %s (this takes a minute)", archive.name, dest)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = Path(member.filename)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(
                        f"{archive.name} contains an unsafe path: {member.filename}"
                    )
            zf.extractall(dest)
    marker.write_text("\n".join(a.name for a in archives))
    return dest


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
