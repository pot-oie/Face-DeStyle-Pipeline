#!/usr/bin/env python3
"""Archive one completed output run, verify it, and optionally remove the run directory."""

import argparse
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_outputs_root(value: Path | None) -> Path:
    if value is not None:
        root = value
    else:
        project_root = os.environ.get("FACE_DESTYLE_ROOT")
        if not project_root:
            raise ValueError("--outputs-root or FACE_DESTYLE_ROOT is required")
        root = Path(project_root) / "outputs"
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"outputs root does not exist: {resolved}")
    return resolved


def validate_run_directory(run_value: Path, outputs_root: Path) -> Path:
    if run_value.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run_value}")
    run_dir = run_value.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    try:
        relative = run_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"run directory must be inside outputs root: {outputs_root}") from exc
    if relative == Path("."):
        raise ValueError("refusing to archive or clean the entire outputs root")
    symlinks = [path for path in run_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"run directory contains symlinks; first entry: {symlinks[0]}")
    return run_dir


def archive_run(run_dir: Path, archive: Path) -> tuple[int, str, Path]:
    archive = archive.expanduser().resolve()
    if archive.suffix.lower() != ".zip":
        raise ValueError("archive path must end in .zip")
    try:
        archive.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError("archive must be outside the run directory")
    checksum_path = archive.with_name(f"{archive.name}.sha256")
    if archive.exists() or checksum_path.exists():
        raise FileExistsError(f"refusing to overwrite archive or checksum: {archive}")

    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"run directory contains no files: {run_dir}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{archive.name}.",
        suffix=".tmp",
        dir=archive.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for path in files:
                relative = path.relative_to(run_dir)
                archive_name = PurePosixPath(run_dir.name, *relative.parts)
                bundle.write(path, arcname=str(archive_name))
        with zipfile.ZipFile(temporary_path) as bundle:
            bad_file = bundle.testzip()
            if bad_file is not None:
                raise RuntimeError(f"ZIP verification failed at member: {bad_file}")
            if len(bundle.infolist()) != len(files):
                raise RuntimeError("ZIP verification found an unexpected member count")
        temporary_path.replace(archive)
    finally:
        temporary_path.unlink(missing_ok=True)

    checksum = file_sha256(archive)
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    return len(files), checksum, checksum_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run below outputs.")
    parser.add_argument("--archive", type=Path, required=True, help="New ZIP path outside run-dir.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        help="Safety boundary; defaults to $FACE_DESTYLE_ROOT/outputs.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove run-dir only after ZIP and checksum verification succeeds.",
    )
    args = parser.parse_args()

    try:
        outputs_root = resolve_outputs_root(args.outputs_root)
        run_dir = validate_run_directory(args.run_dir, outputs_root)
        file_count, checksum, checksum_path = archive_run(run_dir, args.archive)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    archive = args.archive.expanduser().resolve()
    print(f"Verified ZIP: {archive}")
    print(f"Archived files: {file_count}")
    print(f"SHA-256: {checksum}")
    print(f"Checksum file: {checksum_path}")
    if args.cleanup:
        shutil.rmtree(run_dir)
        print(f"Removed generated run directory: {run_dir}")
    else:
        print(f"Kept generated run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
