"""Guards on the dataset adapters.

These exist because of a real incident, the same shape as the one
`test_contract.py` covers: a training run reported success while two of its
five sources contributed almost nothing.

  * PlantWild yielded 0 images. It ships as zips, nothing extracted them, and
    an adapter that finds no class folders returns an empty list rather than
    raising.
  * The hydroponic lettuce map used folder names the dataset does not use
    ("K deficiency" vs the real "K Deficient"), so 147 of 209 images and 3 of
    the 4 vertical classes were dropped.

Both failures were silent. A source contributing zero images looks exactly
like a source that is legitimately empty, so nothing in the pipeline could
tell the difference — hence tests that pin the shapes down directly.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_find_dir_treats_names_as_priority_not_a_set(tmp_path):
    """The exact collision that made PlantWild yield nothing.

    The extraction directory is named `plantwild` and the usable class
    folders live in `plantwild_v2` inside it. Matching "whichever name turns
    up first" returned the wrapper, whose only child is `plantwild_v2` — not
    a class folder, so every image was skipped.
    """
    from ml.sources import _find_dir

    root = tmp_path / "plantwild"
    (root / "plantwild_v2" / "tomato late blight").mkdir(parents=True)

    assert _find_dir(root, "plantwild_v2", "plantwild") == root / "plantwild_v2"
    # Reversing the priority must reverse the answer, or the order is a lie.
    assert _find_dir(root, "plantwild", "plantwild_v2") == root


def test_find_dir_is_deterministic_when_a_name_repeats(tmp_path):
    """Shallowest match wins, so a nested duplicate cannot flip between runs."""
    from ml.sources import _find_dir

    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "nested" / "deeper" / "images").mkdir(parents=True)

    assert _find_dir(root, "images") == root / "images"


def test_lettuce_hydro_map_uses_the_folder_names_that_actually_ship():
    """The vertical domain is ~200 images from one source; a key typo here
    silently deletes most of the deployment domain."""
    from ml.taxonomy import LETTUCE_HYDRO_MAP

    shipped = {"Healthy", "N Deficient", "K Deficient", "Wilt fungal"}
    missing = shipped - set(LETTUCE_HYDRO_MAP)
    assert not missing, f"kagglehub v2 folder names not mapped: {sorted(missing)}"

    # All four vertical classes must be reachable, not just healthy.
    assert len(set(LETTUCE_HYDRO_MAP[k] for k in shipped)) == 4


def test_ensure_extracted_unpacks_zips_and_is_a_noop_for_directories(tmp_path):
    from ml.sources import ensure_extracted

    plain = tmp_path / "plain"
    (plain / "someclass").mkdir(parents=True)
    assert ensure_extracted("plantvillage", plain, tmp_path / "cache") == plain

    src = tmp_path / "zipped"
    src.mkdir()
    with zipfile.ZipFile(src / "data.zip", "w") as zf:
        zf.writestr("data/classA/a.jpg", "x")

    out = ensure_extracted("somesource", src, tmp_path / "cache")
    assert (out / "data" / "classA" / "a.jpg").exists()
    # Second call must not re-extract; the marker is what makes it cheap.
    assert ensure_extracted("somesource", src, tmp_path / "cache") == out


def test_ensure_extracted_refuses_path_traversal(tmp_path):
    """An archive that writes outside its destination is a rejected input,
    not something to unpack and hope about."""
    from ml.sources import ensure_extracted

    src = tmp_path / "evil"
    src.mkdir()
    with zipfile.ZipFile(src / "evil.zip", "w") as zf:
        zf.writestr("../escaped.txt", "x")

    with pytest.raises(ValueError, match="unsafe path"):
        ensure_extracted("evil", src, tmp_path / "cache")


def test_preferred_archive_is_enforced_when_a_source_ships_several():
    """PlantWild ships both v1 and v2; picking the wrong one yields a flat
    `images/` dump with no recoverable labels."""
    from ml.sources import SOURCE_ARCHIVES

    assert SOURCE_ARCHIVES["plantwild"] == "plantwild_v2.zip"
