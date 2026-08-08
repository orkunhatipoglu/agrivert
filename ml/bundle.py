"""Package and fetch trained model versions.

The point of this module: **training happens once, on one machine.** Everyone
else downloads the result. Weights are ~26MB of binary, which git rejects and
which nobody should have to reproduce with a GPU and 8GB of datasets.

A bundle is a `.tar.gz` containing exactly the artifacts serving needs plus a
`MANIFEST.json` with a SHA-256 per file. Hashes are checked on extract, so a
truncated download fails loudly instead of producing a model that loads and
predicts garbage.

    # on the training machine
    python -m ml.bundle pack artifacts/ --version v2-vertical-20260809

    # on any other machine
    python -m ml.bundle fetch https://.../v2-vertical-20260809.tar.gz \\
        --into backend/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ml.contract import REQUIRED_ARTIFACTS

MANIFEST_NAME = "MANIFEST.json"
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(source: Path, version: str) -> dict:
    """Manifest describing a model version: hashes plus enough metrics to tell
    two bundles apart without extracting them."""
    metadata = json.loads((source / "metadata.json").read_text())
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata.get("model_name"),
        "architecture": metadata.get("architecture"),
        "num_classes": metadata.get("num_classes"),
        "classes": metadata.get("classes", []),
        "metrics": metadata.get("metrics", {}),
        "files": {
            name: {
                "sha256": sha256_file(source / name),
                "bytes": (source / name).stat().st_size,
            }
            for name in REQUIRED_ARTIFACTS
        },
    }


def pack(source: Path, version: str, out_dir: Path) -> Path:
    missing = [f for f in REQUIRED_ARTIFACTS if not (source / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source} is missing required artifact(s): {', '.join(missing)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(source, version)

    bundle_path = out_dir / f"{version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / version
        staging.mkdir()
        for name in REQUIRED_ARTIFACTS:
            shutil.copy2(source / name, staging / name)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(staging, arcname=version)

    # Sidecar hash so a fetch can verify the archive itself, before it trusts
    # anything inside it.
    digest = sha256_file(bundle_path)
    (out_dir / f"{version}.tar.gz.sha256").write_text(f"{digest}  {version}.tar.gz\n")
    (out_dir / f"{version}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return bundle_path


def _download(url: str, dest: Path, quiet: bool = False) -> None:
    def report(count, block, total):
        if quiet or total <= 0:
            return
        pct = min(100, count * block * 100 // total)
        sys.stdout.write(f"\r  downloading… {pct}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report)  # noqa: S310
    if not quiet:
        sys.stdout.write("\r  downloading… done\n")


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing entries that escape the destination.

    A bundle is downloaded from a URL, so it is untrusted input; a crafted
    archive with `../` members would otherwise write anywhere on disk.
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive contains a link, refusing: {member.name}")
    tar.extractall(dest)  # noqa: S202 - members validated above


def fetch(
    source: str,
    into: Path,
    expect_sha256: str | None = None,
    version: str | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Install a bundle from a URL or local path into a model registry dir."""
    into.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "bundle.tar.gz"

        if source.startswith(("http://", "https://")):
            if not quiet:
                print(f"fetching {source}")
            _download(source, archive, quiet)
        else:
            local = Path(source)
            if not local.is_file():
                raise FileNotFoundError(f"no such bundle: {local}")
            shutil.copy2(local, archive)

        actual = sha256_file(archive)
        if expect_sha256:
            if actual.lower() != expect_sha256.strip().lower():
                raise ValueError(
                    "bundle SHA-256 mismatch — the download is corrupt or has "
                    f"been altered.\n  expected {expect_sha256}\n  actual   {actual}"
                )
            if not quiet:
                print("  archive hash verified")
        elif not quiet:
            print(f"  archive sha256 {actual}  (no --sha256 given, not verified)")

        extract_root = tmp_path / "x"
        extract_root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, extract_root)

        roots = [p for p in extract_root.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one directory in the bundle, found {len(roots)}"
            )
        staged = roots[0]
        resolved_version = version or staged.name

        manifest_path = staged / MANIFEST_NAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            for name, info in manifest.get("files", {}).items():
                got = sha256_file(staged / name)
                if got != info["sha256"]:
                    raise ValueError(
                        f"{name} failed its manifest hash check — bundle is corrupt"
                    )
            if not quiet:
                print(f"  {len(manifest.get('files', {}))} file hashes verified")
        elif not quiet:
            print("  warning: bundle has no MANIFEST.json; skipping file hash checks")

        missing = [f for f in REQUIRED_ARTIFACTS if not (staged / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"bundle is missing required artifact(s): {', '.join(missing)}"
            )

        dest = into / resolved_version
        if dest.exists():
            if not force:
                raise FileExistsError(
                    f"{dest} already exists; pass --force to replace it"
                )
            shutil.rmtree(dest)
        shutil.move(str(staged), str(dest))

    if not quiet:
        print(f"\ninstalled model version {resolved_version} -> {dest}")
    return dest


def _cmd_pack(args) -> int:
    path = pack(args.artifacts.resolve(), args.version, args.out.resolve())
    manifest = json.loads((args.out / f"{args.version}.manifest.json").read_text())
    print(f"packed {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {sha256_file(path)}")
    field = manifest.get("metrics", {}).get("test_field", {})
    if field:
        print(f"test_field accuracy {field.get('accuracy')}")
    print(
        "\nUpload the .tar.gz (e.g. as a GitHub Release asset), then others run:\n"
        f"  python -m ml.bundle fetch <url> --into backend/models --sha256 {sha256_file(path)}"
    )
    return 0


def _cmd_fetch(args) -> int:
    try:
        fetch(
            args.source,
            args.into.resolve(),
            expect_sha256=args.sha256,
            version=args.version,
            force=args.force,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ml.bundle", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="Package an artifacts dir for distribution")
    p.add_argument("artifacts", type=Path)
    p.add_argument("--version", required=True)
    p.add_argument("--out", type=Path, default=Path("dist"))
    p.set_defaults(func=_cmd_pack)

    f = sub.add_parser("fetch", help="Install a bundle from a URL or file")
    f.add_argument("source", help="URL or local .tar.gz path")
    f.add_argument("--into", type=Path, default=Path("backend/models"))
    f.add_argument("--sha256", default=None, help="Expected archive hash")
    f.add_argument("--version", default=None, help="Override the installed name")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=_cmd_fetch)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
