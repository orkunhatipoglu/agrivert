"""Near-duplicate detection, so a split measures generalisation.

The hydroponic lettuce set — the only true vertical source, and the one every
selection and calibration decision rests on — is burst photography. Filenames
like `IMG20251222130237.jpg` and `IMG20251222130242.jpg` are the same plant
five seconds apart. Between 12%% and 40%% of images per class have a near
twin.

Split those at random and the val set is largely re-photographs of the train
set. The model then scores ~0.96 F1 on vertical by epoch 1 while having
learned nothing transferable, and — worse — the confidence threshold gets
fitted against those inflated scores, so serving accepts predictions it
should have refused.

The fix is to split by *group* rather than by image: near-duplicates are
merged into one unit that lands wholly in train, or wholly in val, never
across. The reported number then answers the question that matters, which is
"how does this do on a plant it has never seen", not "can it recognise a
photo it has already memorised".

Standalone leakage report:

    python -m ml.dedup /path/to/dataset --threshold 5
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 8x8 difference hash -> 64 bits. Small enough to compare a whole dataset
# pairwise with numpy, robust to the resize/recompress differences that make
# byte-level hashing useless here.
HASH_SIZE = 8

# Hamming distance at or below this counts as "the same subject". 5/64 is
# deliberately conservative: it catches burst frames and re-crops while
# leaving genuinely different plants of the same disease apart.
DEFAULT_THRESHOLD = 5

_CACHE_VERSION = 1


def dhash(path: Path, size: int = HASH_SIZE) -> int | None:
    """Row-wise difference hash. None if the image cannot be read."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size))
            px = np.asarray(im, dtype=np.int16)
    except Exception:
        return None

    bits = px[:, 1:] > px[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _hash_one(path_str: str):
    return path_str, dhash(Path(path_str))


def compute_hashes(
    paths: Sequence[Path], cache_file: Path | None = None, workers: int = 4
) -> dict[str, int]:
    """Hash every path, reusing a cache keyed by (size, mtime).

    Hashing 27k images is a couple of minutes; doing it on every run would
    make `--dry-run` useless as a quick check, hence the cache.
    """
    cache: dict[str, list] = {}
    if cache_file and cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text())
            if blob.get("version") == _CACHE_VERSION:
                cache = blob.get("entries", {})
        except (json.JSONDecodeError, OSError):
            log.warning("dedup cache unreadable, rebuilding")

    out: dict[str, int] = {}
    todo: list[str] = []
    for p in paths:
        key = str(p)
        try:
            stat = p.stat()
        except OSError:
            continue
        hit = cache.get(key)
        if hit and hit[0] == stat.st_size and hit[1] == int(stat.st_mtime):
            out[key] = hit[2]
        else:
            todo.append(key)

    if todo:
        log.info("hashing %d image(s) for near-duplicate detection…", len(todo))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for key, value in pool.map(_hash_one, todo, chunksize=64):
                if value is not None:
                    out[key] = value

    if cache_file:
        entries = {}
        for key, value in out.items():
            try:
                stat = Path(key).stat()
            except OSError:
                continue
            entries[key] = [stat.st_size, int(stat.st_mtime), value]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"version": _CACHE_VERSION, "entries": entries})
        )
    return out


def _pairs_within(values: np.ndarray, threshold: int) -> Iterable[tuple[int, int]]:
    """Indices of pairs closer than `threshold` bits, via chunked popcount."""
    n = len(values)
    if n < 2:
        return
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = values[start:stop, None] ^ values[None, :]
        dist = np.bitwise_count(block)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:  # each unordered pair once, and never self
                yield i, j


def group_duplicates(
    hashes: Sequence[int | None], threshold: int = DEFAULT_THRESHOLD
) -> list[int]:
    """Union-find over near-duplicate pairs; returns a group id per input.

    An unhashable image becomes its own group — excluding it would silently
    drop data, and keeping it ungrouped is the conservative choice.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    usable = [i for i, h in enumerate(hashes) if h is not None]
    if usable:
        values = np.array([hashes[i] for i in usable], dtype=np.uint64)
        for i, j in _pairs_within(values, threshold):
            union(usable[i], usable[j])

    remap: dict[int, int] = {}
    out = []
    for i in range(n):
        root = find(i)
        out.append(remap.setdefault(root, len(remap)))
    return out


def group_within_buckets(
    bucket_keys: Sequence[Hashable],
    hashes: Sequence[int | None],
    threshold: int = DEFAULT_THRESHOLD,
) -> list[int]:
    """Group near-duplicates independently inside each bucket.

    Comparing only within a bucket keeps the pairwise cost tractable — the
    largest class holds ~4.5k images, against ~27k for the whole corpus — and
    it matches what the split actually needs. Leakage is only harmful between
    the train and val halves of the *same* (label, domain) bucket; two
    identical-looking images filed under different labels are a labelling
    problem, not a leakage one, and merging them here would hide that.
    """
    by_bucket: dict[Hashable, list[int]] = defaultdict(list)
    for i, key in enumerate(bucket_keys):
        by_bucket[key].append(i)

    out = [0] * len(bucket_keys)
    next_id = 0
    for key in sorted(by_bucket, key=repr):
        idxs = by_bucket[key]
        local = group_duplicates([hashes[i] for i in idxs], threshold)
        remap: dict[int, int] = {}
        for i, gid in zip(idxs, local):
            if gid not in remap:
                remap[gid] = next_id
                next_id += 1
            out[i] = remap[gid]
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Report near-duplicate leakage")
    ap.add_argument("root", type=Path)
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in sorted(args.root.rglob("*")) if p.suffix.lower() in exts]
    if not paths:
        print(f"no images under {args.root}")
        return 1

    hashes = compute_hashes(paths, workers=args.workers)
    values = [hashes.get(str(p)) for p in paths]
    groups = group_duplicates(values, args.threshold)
    n_groups = len(set(groups))
    print(f"{len(paths)} images -> {n_groups} distinct groups")
    print(
        f"{len(paths) - n_groups} image(s) are near-duplicates of another "
        f"({(len(paths) - n_groups) / len(paths):.0%} of the set)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
