#!/usr/bin/env python3
"""Write one read-only inventory of code, data, models, runs, and archives."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.models import ModelRegistry

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
PACKAGE_NAMES = (
    "face-destyle-pipeline",
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "onnxruntime-gpu",
    "insightface",
)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def run_text(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def read_jsonl_loose(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{type(exc).__name__}: {exc}"]
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("JSONL row is not an object")
            rows.append(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"line {number}: {type(exc).__name__}: {exc}")
    return rows, errors


def formal_manifest_inventory(
    manifest: Path,
    data_root: Path,
    *,
    verify_checksums: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    rows, errors = read_jsonl_loose(manifest)
    source_splits = {
        str(row["source_id"]): str(row["split"])
        for row in rows
        if row.get("source_id") and row.get("split")
    }
    payload: dict[str, Any] = {
        "path": str(manifest.resolve()),
        "records": len(rows),
        "parse_errors": errors,
        "by_split": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "by_split_style": dict(
            sorted(
                Counter(
                    f"{row.get('split')}:{row.get('style_category')}" for row in rows
                ).items()
            )
        ),
        "unique_source_ids": len(source_splits),
        "unique_source_groups": len(
            {str(row.get("source_group_id")) for row in rows if row.get("source_group_id")}
        ),
        "unique_sha256": len({str(row.get("sha256")) for row in rows if row.get("sha256")}),
        "checksum_validation": "not_requested",
    }
    if verify_checksums:
        try:
            validated = load_dataset_manifest(manifest, data_root=data_root)
            payload["checksum_validation"] = "passed"
            payload["checksum_validated_records"] = len(validated)
        except Exception as exc:  # noqa: BLE001 - inventory must preserve every audit failure
            payload["checksum_validation"] = "failed"
            payload["checksum_error"] = f"{type(exc).__name__}: {exc}"
    return payload, source_splits


def scan_run(records_path: Path, root: Path, source_splits: dict[str, str]) -> dict[str, Any]:
    rows, errors = read_jsonl_loose(records_path)
    run_dir = records_path.parent
    source_ids = [str(row.get("source_id")) for row in rows if row.get("source_id")]
    output_paths = [Path(str(row["output_path"])) for row in rows if row.get("output_path")]
    failure_path = run_dir / "failures.jsonl"
    failures, failure_errors = (
        read_jsonl_loose(failure_path) if failure_path.is_file() else ([], [])
    )
    image_dir = run_dir / "images"
    image_count = (
        sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in image_dir.rglob("*")
        )
        if image_dir.is_dir()
        else 0
    )
    return {
        "run_dir": relative_or_absolute(run_dir, root),
        "records_path": relative_or_absolute(records_path, root),
        "records": len(rows),
        "record_parse_errors": errors,
        "unique_source_ids": len(set(source_ids)),
        "formal_split_coverage": dict(
            sorted(
                Counter(
                    source_splits.get(source_id, "not_in_formal_manifest")
                    for source_id in source_ids
                ).items()
            )
        ),
        "backends": dict(sorted(Counter(str(row.get("backend")) for row in rows).items())),
        "seeds": dict(sorted(Counter(str(row.get("seed")) for row in rows).items())),
        "styles": dict(sorted(Counter(str(row.get("style_category")) for row in rows).items())),
        "recorded_outputs": len(output_paths),
        "recorded_outputs_present": sum(path.is_file() for path in output_paths),
        "image_files_in_images_dir": image_count,
        "failures": len(failures),
        "failure_parse_errors": failure_errors,
    }


def scan_data_area(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path.resolve()),
        "exists": True,
        "files": len(files),
        "image_files": sum(item.suffix.lower() in IMAGE_SUFFIXES for item in files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def scan_archives(root: Path) -> list[dict[str, Any]]:
    archives = []
    searched: set[Path] = set()
    for name in ("archives", "packages", "transfers"):
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.zip")):
            resolved = path.resolve()
            if resolved in searched:
                continue
            searched.add(resolved)
            checksum = path.with_name(f"{path.name}.sha256")
            archives.append(
                {
                    "path": relative_or_absolute(path, root),
                    "bytes": path.stat().st_size,
                    "checksum_sidecar": relative_or_absolute(checksum, root)
                    if checksum.is_file()
                    else None,
                }
            )
    return archives


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("FACE_DESTYLE_ROOT", ".")),
        help="Persistent Face-DeStyle root; defaults to FACE_DESTYLE_ROOT.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "External Face-DeStyle-Data root; defaults to FACE_DESTYLE_DATA_ROOT "
            "or ROOT/data/Face-DeStyle-Data."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-data-checksums", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root
        or (
            Path(os.environ["FACE_DESTYLE_DATA_ROOT"])
            if os.environ.get("FACE_DESTYLE_DATA_ROOT")
            else None
        )
        or (root / "data" / "Face-DeStyle-Data")
    ).expanduser().resolve()
    manifest = repo_root / "data" / "manifests" / "formal-v1" / "inputs.jsonl"
    if manifest.is_file():
        manifest_payload, source_splits = formal_manifest_inventory(
            manifest,
            data_root,
            verify_checksums=args.verify_data_checksums,
        )
    else:
        manifest_payload = {"path": str(manifest), "exists": False}
        source_splits = {}

    registry = ModelRegistry.from_yaml(repo_root / "configs" / "models.yaml")
    models = []
    for check in registry.check_all(root=root):
        asset = registry.require(check.name)
        models.append(
            {
                "name": check.name,
                "role": asset.role,
                "available": check.available,
                "location": str(check.location) if check.location else None,
                "missing_files": list(check.missing_files),
                "reason": check.reason,
            }
        )

    outputs_root = root / "outputs"
    runs = (
        [
            scan_run(path, root, source_splits)
            for path in sorted(outputs_root.rglob("records.jsonl"))
        ]
        if outputs_root.is_dir()
        else []
    )
    acquisition_dir = root / "models" / "download-manifests"
    acquisition_manifests = (
        [
            {"path": relative_or_absolute(path, root), "bytes": path.stat().st_size}
            for path in sorted(acquisition_dir.glob("*"))
            if path.is_file()
        ]
        if acquisition_dir.is_dir()
        else []
    )
    payload = {
        "schema": "face-destyle-workspace-inventory/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "root": str(root),
        "data_root": str(data_root),
        "repo": {
            "path": str(repo_root),
            "commit": run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "branch": run_text(["git", "branch", "--show-current"], cwd=repo_root),
            "status": (run_text(["git", "status", "--short"], cwd=repo_root) or "").splitlines(),
        },
        "gpu": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "packages": package_versions(),
        "formal_manifest": manifest_payload,
        "data_areas": {
            "pilot": scan_data_area(data_root / "raw"),
            "batch1": scan_data_area(data_root / "batch1"),
            "batch2": scan_data_area(data_root / "batch2"),
        },
        "models": models,
        "model_acquisition_manifests": acquisition_manifests,
        "runs": runs,
        "archives": scan_archives(root),
        "notes": [
            "Model weights and archives were not re-hashed.",
            "Recorded output presence is checked against paths stored in each records.jsonl.",
            "The script is read-only except for writing this inventory JSON.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    available = sum(bool(model["available"]) for model in models)
    print(f"Wrote workspace inventory: {args.output.resolve()}")
    print(f"Formal manifest records: {manifest_payload.get('records', 0)}")
    print(f"Model assets available: {available}/{len(models)}")
    print(f"Run record sets found: {len(runs)}")
    print(f"Archives found: {len(payload['archives'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
