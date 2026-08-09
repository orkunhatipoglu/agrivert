"""
Blended PlantVillage + PlantDoc dataset handling for Agrivert.

WHY THIS MODULE EXISTS
----------------------
PlantVillage is studio photography: one detached leaf, plain uniform
background, even lighting. A CNN trained on it reaches ~99% on its own test
split and then collapses on real farm photos, because a large part of what it
learned is "clean grey background => leaf classification task" rather than the
lesion morphology we actually care about.

PlantDoc is the corrective: ~2.5k real in-field photos, cluttered backgrounds,
uneven light, leaves still attached to plants, taken on phones. It is ~4% the
size of PlantVillage, so blending alone is not enough — it has to be
oversampled (see build_sample_weights) and paired with heavy augmentation
(see build_train_transform), and, most importantly, it has to be *evaluated
separately* so the field number is never hidden inside a studio-dominated
average.

This module has NO torch import and only a lazy albumentations import, so the
scanning / mapping / splitting logic can be unit-tested without a GPU stack.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Sourced from the shared contract so training and serving cannot disagree.
from ml.contract import (  # noqa: E402
    DEFAULT_CENTER_CROP,
    DEFAULT_RESIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_RATIO,
)

IMAGE_SIZE = DEFAULT_CENTER_CROP

DOMAIN_STUDIO = "plantvillage"
DOMAIN_FIELD = "plantdoc"

# --------------------------------------------------------------------------
# Canonical label space
# --------------------------------------------------------------------------
# The 38 PlantVillage class folders, exactly as they appear on disk. Kept here
# as a sanity check: if a scan produces something different, the wrong dataset
# (or the wrong variant folder) has been pointed at.
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# --------------------------------------------------------------------------
# PlantDoc -> PlantVillage class mapping
# --------------------------------------------------------------------------
# PlantDoc folder names are free-form and use "<Crop> leaf" for the healthy
# class. All 28 PlantDoc classes map onto a PlantVillage class; the mapping was
# built against the actual folder listing of manojkumarcs28/plantdoc-dataset
# (28 train folders, 27 test folders).
#
# Watch out for the two easy mistakes in here:
#   * "Corn leaf blight" is Northern Leaf Blight, NOT Cercospora/Gray leaf spot
#     ("Corn Gray leaf spot" is the separate Cercospora class).
#   * "Apple rust leaf" is Cedar apple rust, not Black rot.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    # Seen in some PlantDoc redistributions; harmless if absent.
    "Potato leaf": "Potato___healthy",
}

# PlantVillage classes with no PlantDoc counterpart. These stay studio-only:
# the model will have seen zero field examples of them, so their field
# behaviour is unvalidated. Surfaced in metadata.json rather than left implicit.
#
# NOTE: this constant is the best case, assuming every mapping key exists on
# disk. The "Potato leaf" folder is absent from manojkumarcs28/plantdoc-dataset,
# so the real count there is 10, not 9. build_blend() recomputes this from the
# folders actually found and that recomputed list is what reaches metadata.json.
STUDIO_ONLY_CLASSES = sorted(
    set(PLANTVILLAGE_CLASSES) - set(PLANTDOC_TO_PLANTVILLAGE.values())
)


def validate_mapping() -> None:
    """Fail loudly if the mapping drifts out of sync with the label space."""
    unknown = sorted(set(PLANTDOC_TO_PLANTVILLAGE.values()) - set(PLANTVILLAGE_CLASSES))
    if unknown:
        raise ValueError(f"Mapping targets not in PlantVillage label space: {unknown}")
    if len(set(PLANTVILLAGE_CLASSES)) != len(PLANTVILLAGE_CLASSES):
        raise ValueError("Duplicate entries in PLANTVILLAGE_CLASSES")


# --------------------------------------------------------------------------
# Filesystem scanning
# --------------------------------------------------------------------------
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")


def scan_class_folders(root: Path) -> dict[str, list[Path]]:
    """Return {folder_name: [image paths]} for an ImageFolder-style directory."""
    root = Path(root)
    out: dict[str, list[Path]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        imgs = sorted(p for p in entry.rglob("*") if p.is_file() and _is_image(p))
        if imgs:
            out[entry.name] = imgs
    return out


def find_subdir(start: Path, name: str, require_class_dirs: bool = True) -> Path:
    """Locate a named directory under `start`, however the archive was extracted.

    kagglehub extracts PlantVillage to a nested 'plantvillage dataset/' folder
    (with a space) and PlantDoc to a nested 'PlantDoc/' folder, so nothing here
    assumes a fixed depth.
    """
    start = Path(start).expanduser().resolve()
    if start.name.lower() == name.lower() and start.is_dir():
        return start
    direct = start / name
    if direct.is_dir():
        return direct
    candidates = [
        p for p in sorted(start.rglob("*"))
        if p.is_dir() and p.name.lower() == name.lower()
    ]
    for cand in candidates:
        if not require_class_dirs:
            return cand
        if any(c.is_dir() for c in cand.iterdir()):
            return cand
    raise FileNotFoundError(
        f"Could not find a '{name}' directory under {start}. "
        f"Pass an explicit --plantvillage-root / --plantdoc-root."
    )


def load_plantvillage(root: Path, variant: str = "color") -> list[tuple[Path, str, str]]:
    """-> [(path, plantvillage_class, domain)] for the studio dataset."""
    color_dir = find_subdir(Path(root), variant)
    folders = scan_class_folders(color_dir)
    if not folders:
        raise FileNotFoundError(f"No class folders with images under {color_dir}")
    unexpected = sorted(set(folders) - set(PLANTVILLAGE_CLASSES))
    if unexpected:
        raise ValueError(
            f"Unexpected PlantVillage class folders under {color_dir}: {unexpected[:5]}"
            " — is this really the PlantVillage 'color' directory?"
        )
    return [(p, cls, DOMAIN_STUDIO) for cls, paths in folders.items() for p in paths]


def load_plantdoc(root: Path) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]], dict]:
    """-> (train_samples, test_samples, report) for the field dataset.

    PlantDoc ships its own train/ and test/ split. We keep the official test
    split completely out of training: it becomes the held-out FIELD test set,
    which is the only number in this project that predicts production
    behaviour.
    """
    base = find_subdir(Path(root), "PlantDoc")
    train_dir, test_dir = base / "train", base / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {base}/train and {base}/test. Found: "
            f"{sorted(p.name for p in base.iterdir() if p.is_dir())}"
        )

    report: dict = {"unmapped_folders": [], "mapped": {}, "counts": {}}
    out: dict[str, list[tuple[Path, str, str]]] = {}
    for split, d in (("train", train_dir), ("test", test_dir)):
        samples = []
        for folder, paths in scan_class_folders(d).items():
            target = PLANTDOC_TO_PLANTVILLAGE.get(folder.strip())
            if target is None:
                report["unmapped_folders"].append(f"{split}/{folder}")
                continue
            report["mapped"][folder] = target
            samples.extend((p, target, DOMAIN_FIELD) for p in paths)
        out[split] = samples
        report["counts"][split] = len(samples)
    return out["train"], out["test"], report


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(labels, fracs: tuple[float, ...], seed: int) -> list[list[int]]:
    """Per-class split into len(fracs)+1 parts; the remainder goes to part 0.

    Guarantees part 0 (train) keeps at least one sample of every class, which
    matters because PlantDoc's rarest class has only 2 images.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    parts: list[list[int]] = [[] for _ in range(len(fracs) + 1)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        take = []
        for f in fracs:
            k = int(round(n * f))
            # never starve the train part
            if n - sum(take) - k < 1:
                k = max(0, n - sum(take) - 1)
            take.append(k)
        cursor = 0
        for part_i, k in enumerate(take, start=1):
            parts[part_i].extend(idx[cursor:cursor + k].tolist())
            cursor += k
        parts[0].extend(idx[cursor:].tolist())
    for p in parts:
        rng.shuffle(p)
    return parts


def build_sample_weights(labels, domains, field_oversample: float,
                         balance_classes: bool = True) -> np.ndarray:
    """Sampling weight per training sample.

    Two separate corrections multiplied together:
      * class imbalance — PlantVillage spans 152..5507 images per class (36x)
      * domain imbalance — field data is ~4% of the blend by count. At
        field_oversample=8 it becomes roughly a third of what the model sees
        per epoch, which is the point of the exercise.
    """
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    counts = Counter(labels.tolist())
    if balance_classes:
        w = np.array([1.0 / counts[l] for l in labels], dtype=np.float64)
    else:
        w = np.ones(len(labels), dtype=np.float64)
    w[domains == DOMAIN_FIELD] *= float(field_oversample)
    return w


def effective_domain_share(weights, domains) -> float:
    """Expected fraction of drawn samples that are field images."""
    weights = np.asarray(weights, dtype=np.float64)
    domains = np.asarray(domains)
    total = weights.sum()
    return float(weights[domains == DOMAIN_FIELD].sum() / total) if total else 0.0


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def _try_build(label: str, *candidates):
    """Construct the first candidate that the installed albumentations accepts.

    albumentations renamed several constructor arguments between 1.x and 2.x
    (RandomResizedCrop height/width -> size, CoarseDropout max_holes ->
    num_holes_range). Rather than pin a single version and break on the other,
    each transform is attempted in order and the first one that constructs wins.
    """
    errors = []
    for factory in candidates:
        try:
            return factory()
        except (TypeError, ValueError, AttributeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not construct '{label}' with the installed albumentations. "
        f"Attempts: {errors}"
    )


def build_train_transform(image_size: int = IMAGE_SIZE, strength: float = 1.0):
    """Heavy augmentation aimed squarely at background reliance.

    The ordering matters: geometry first (so crops/rotations happen on full
    resolution), then photometric, then blur/noise, then dropout last so holes
    are punched into the final composed view.

    CoarseDropout is the single most on-point transform here — randomly erasing
    patches forces the network to classify from whatever leaf tissue survives
    rather than memorising a global background signature.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    s = float(strength)
    rrc = _try_build(
        "RandomResizedCrop",
        lambda: A.RandomResizedCrop(size=(image_size, image_size),
                                    scale=(0.4, 1.0), ratio=(0.75, 1.33), p=1.0),
        lambda: A.RandomResizedCrop(height=image_size, width=image_size,
                                    scale=(0.4, 1.0), ratio=(0.75, 1.33), p=1.0),
    )
    ssr = _try_build(
        "ShiftScaleRotate",
        lambda: A.ShiftScaleRotate(shift_limit=0.0625 * s, scale_limit=0.2 * s,
                                   rotate_limit=int(30 * s), border_mode=0, p=0.7),
        lambda: A.Affine(translate_percent=0.0625 * s, scale=(1 - 0.2 * s, 1 + 0.2 * s),
                         rotate=(-30 * s, 30 * s), p=0.7),
    )
    coarse = _try_build(
        "CoarseDropout",
        lambda: A.CoarseDropout(num_holes_range=(1, 8),
                                hole_height_range=(0.05, 0.18),
                                hole_width_range=(0.05, 0.18), p=0.5),
        lambda: A.CoarseDropout(max_holes=8, min_holes=1,
                                max_height=int(0.18 * image_size),
                                max_width=int(0.18 * image_size),
                                min_height=int(0.05 * image_size),
                                min_width=int(0.05 * image_size), p=0.5),
        lambda: A.Cutout(num_holes=8, max_h_size=int(0.18 * image_size),
                         max_w_size=int(0.18 * image_size), p=0.5),
    )
    color = _try_build(
        "ColorJitter",
        lambda: A.ColorJitter(brightness=0.35 * s, contrast=0.35 * s,
                              saturation=0.35 * s, hue=0.06 * s, p=0.8),
    )
    return A.Compose([
        rrc,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        ssr,
        color,
        A.RandomBrightnessContrast(brightness_limit=0.3 * s,
                                   contrast_limit=0.3 * s, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.Defocus(radius=(1, 4)),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(),
            A.ImageCompression(),
        ], p=0.25),
        coarse,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Deterministic eval preprocessing. predict.py must mirror this exactly."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    resize_to = (
        DEFAULT_RESIZE
        if image_size == DEFAULT_CENTER_CROP
        else int(round(image_size * RESIZE_RATIO))
    )
    return A.Compose([
        A.SmallestMaxSize(max_size=resize_to),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_tta_transforms(image_size: int = IMAGE_SIZE, ratios=(1.0, 1.14, 1.35)):
    """Multi-scale views for test-time averaging. Returns [(transform, flip)].

    Eval keeps only the centre square: the short side is resized to
    `image_size * ratio` and then centre-cropped, so a 4:3 photo loses ~42% of
    its area and a 16:9 photo ~57%. Which pixels survive therefore depends on
    the framing, and re-cropping a photo by a few percent slides that window —
    on held-out field photos, 31% of them changed predicted class under a <=10%
    crop, with confidence swinging 0.15 on average.

    Averaging over several zoom levels shrinks that: flips fall to ~20% and the
    mean swing to ~0.11. Be clear about what this does and does not buy —
    it costs `len(ratios)`x inference and does **not** improve accuracy (on
    val it measured slightly worse, 58.4% vs 60.8%). It buys stability only.
    The instability itself is mostly the model being ~55% accurate on field
    photos: near the decision boundary, small input changes flip the argmax.
    """
    return [(build_eval_transform_at(image_size, r), False) for r in ratios]


def build_eval_transform_at(image_size: int, ratio: float):
    """Eval transform at an explicit resize ratio (see build_tta_transforms)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.SmallestMaxSize(max_size=int(round(image_size * ratio))),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def describe_transform(tf) -> list[str]:
    """Flat list of the transform class names actually constructed."""
    names = []
    for t in getattr(tf, "transforms", []):
        inner = getattr(t, "transforms", None)
        if inner:
            names.append(f"{type(t).__name__}[{', '.join(type(i).__name__ for i in inner)}]")
        else:
            names.append(type(t).__name__)
    return names


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def load_image_rgb(path) -> np.ndarray:
    """Decode to an HWC uint8 RGB array, honouring EXIF rotation.

    PIL is used rather than cv2.imread because phone photos carry EXIF
    orientation (cv2 ignores it, so a sideways field photo would be fed in
    rotated 90 degrees) and because PIL gives clearer errors on truncated files.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


class BlendedDataset:
    """Map-style dataset over (path, class, domain) triples.

    Not a subclass of torch.utils.data.Dataset on purpose — DataLoader only
    needs __len__/__getitem__, and staying torch-free keeps this module (and
    its tests) importable without the GPU stack installed.
    """

    def __init__(self, samples, class_to_idx, transform, return_domain: bool = True):
        self.samples = list(samples)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.return_domain = return_domain
        self.domain_to_idx = {DOMAIN_STUDIO: 0, DOMAIN_FIELD: 1}
        self.targets = [self.class_to_idx[c] for _, c, _ in self.samples]
        self.domains = [d for _, _, d in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, cls, domain = self.samples[i]
        try:
            image = load_image_rgb(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read image {path}: {exc}") from exc
        image = self.transform(image=image)["image"]
        target = self.class_to_idx[cls]
        if self.return_domain:
            return image, target, self.domain_to_idx[domain]
        return image, target


def filter_unreadable(samples, cache_path=None, log=print):
    """Drop images that cannot be decoded.

    PlantDoc contains a handful of truncated / mislabelled-extension files;
    hitting one 40 minutes into a run is a bad way to find out. Results are
    cached because a full pass over ~57k files is not free.

    Note this does a FULL decode, not Image.verify(). verify() only sanity
    checks headers and happily passes a JPEG truncated to a third of its
    length — which then raises inside the DataLoader mid-epoch, i.e. exactly
    the failure this function exists to prevent. Expect a few minutes on the
    first pass; results are cached to `cache_path`.
    """
    from PIL import Image

    cache: dict[str, bool] = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:  # noqa: BLE001
            cache = {}

    good, bad = [], []
    for sample in samples:
        key = str(sample[0])
        ok = cache.get(key)
        if ok is None:
            try:
                with Image.open(key) as im:
                    im.convert("RGB").load()   # full decode, not just headers
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            cache[key] = ok
        (good if ok else bad).append(sample)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    if bad:
        log(f"Dropped {len(bad)} unreadable image(s), e.g. {bad[0][0]}")
    return good, bad


# --------------------------------------------------------------------------
# Blend assembly
# --------------------------------------------------------------------------
def build_blend(plantvillage_root, plantdoc_root, seed: int = 42,
                pv_val_frac: float = 0.1, pv_test_frac: float = 0.1,
                pd_val_frac: float = 0.15, variant: str = "color",
                restrict_to_field_classes: bool = False, log=print) -> dict:
    """Assemble the blended splits.

    Split policy, and the reasoning behind it:
      PlantVillage -> train / val / test        (stratified 80/10/10)
      PlantDoc train -> train / val             (stratified 85/15)
      PlantDoc test  -> FIELD TEST, never trained on, never selected on

    Checkpoint selection uses the *field* validation macro-F1, not the blended
    one. Selecting on a blend dominated by 5.4k studio images would just pick
    whichever epoch happened to overfit PlantVillage hardest, which is the
    exact failure this rewrite exists to fix.
    """
    validate_mapping()

    pv = load_plantvillage(plantvillage_root, variant)
    log(f"PlantVillage: {len(pv)} images across "
        f"{len({c for _, c, _ in pv})} classes")

    pd_train_all, pd_test, pd_report = load_plantdoc(plantdoc_root)
    log(f"PlantDoc: {len(pd_train_all)} train + {len(pd_test)} test images across "
        f"{len({c for _, c, _ in pd_train_all})} mapped classes")
    if pd_report["unmapped_folders"]:
        log(f"  WARNING unmapped PlantDoc folders (ignored): "
            f"{pd_report['unmapped_folders']}")

    field_classes = sorted({c for _, c, _ in pd_train_all} | {c for _, c, _ in pd_test})
    classes = field_classes if restrict_to_field_classes else sorted(
        {c for _, c, _ in pv} | set(field_classes)
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if restrict_to_field_classes:
        pv = [s for s in pv if s[1] in class_to_idx]
        log(f"Restricted to the {len(classes)} field-covered classes; "
            f"PlantVillage reduced to {len(pv)} images")

    pv_train_i, pv_val_i, pv_test_i = stratified_split(
        [c for _, c, _ in pv], (pv_val_frac, pv_test_frac), seed)
    pd_train_i, pd_val_i = stratified_split(
        [c for _, c, _ in pd_train_all], (pd_val_frac,), seed)

    pick = lambda src, idx: [src[i] for i in idx]
    splits = {
        "train": pick(pv, pv_train_i) + pick(pd_train_all, pd_train_i),
        "val_studio": pick(pv, pv_val_i),
        "val_field": pick(pd_train_all, pd_val_i),
        "test_studio": pick(pv, pv_test_i),
        "test_field": list(pd_test),
    }
    splits["val_blended"] = splits["val_studio"] + splits["val_field"]

    missing_in_field = sorted(set(classes) - set(field_classes))
    log("Split sizes: " + " | ".join(f"{k} {len(v)}" for k, v in splits.items()))
    log(f"Label space: {len(classes)} classes "
        f"({len(field_classes)} with field coverage, "
        f"{len(missing_in_field)} studio-only)")

    return {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "splits": splits,
        "field_classes": field_classes,
        "studio_only_classes": missing_in_field,
        "plantdoc_report": pd_report,
    }


def split_summary(samples) -> dict:
    """Per-domain and per-class counts, for logging and metadata.json."""
    by_domain = Counter(d for _, _, d in samples)
    by_class = defaultdict(Counter)
    for _, c, d in samples:
        by_class[c][d] += 1
    return {
        "total": len(samples),
        "by_domain": dict(by_domain),
        "by_class": {c: dict(v) for c, v in sorted(by_class.items())},
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
