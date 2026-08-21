#!/usr/bin/env python3
"""Machine-validate the sealed formal-v1 FLUX test ZIP without displaying images."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

from face_destyle.data.metadata import read_jsonl
from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.schemas import DatasetManifestRecord, DestylizationRecord
from face_destyle.utils.io import load_yaml

BACKEND = "flux1_kontext_dev_prompt_edit_bf16_offloaded"
STYLES = ("3d_cartoon", "comic", "ink", "watercolor")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sidecar_digest(path: Path, archive: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != archive.name:
        raise ValueError("SHA-256 sidecar must contain DIGEST and the exact archive basename")
    digest = fields[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("invalid SHA-256 digest in sidecar")
    return digest


def unique_member(names: list[str], basename: str) -> str | None:
    matches = [name for name in names if PurePosixPath(name).name == basename]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"archive contains multiple {basename} files")
    return matches[0]


def parse_jsonl_bytes(payload: bytes, label: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {label} line {line_number}") from exc
    return rows


def validate_archive(
    archive: Path,
    sidecar: Path,
    manifest: Path,
    styles_config: Path,
) -> dict:
    expected_digest = expected_sidecar_digest(sidecar, archive)
    actual_digest = file_sha256(archive)
    if actual_digest != expected_digest:
        raise ValueError(f"archive SHA-256 mismatch: {actual_digest} != {expected_digest}")

    manifest_rows = read_jsonl(manifest, DatasetManifestRecord)
    test_rows = {row.source_id: row for row in manifest_rows if row.split == "test"}
    sealed_ids = {row.source_id for row in manifest_rows if row.split != "test"}
    if len(test_rows) != 60:
        raise ValueError(f"formal manifest has {len(test_rows)} test sources, expected 60")
    if Counter(row.style_category for row in test_rows.values()) != Counter(
        {style: 15 for style in STYLES}
    ):
        raise ValueError("formal test split is not balanced 15 per frozen style")
    prompts = load_yaml(styles_config)

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC failed for {corrupt}")
        records_member = unique_member(names, "records.jsonl")
        if records_member is None:
            raise ValueError("archive lacks records.jsonl")
        failures_member = unique_member(names, "failures.jsonl")
        record_payloads = parse_jsonl_bytes(bundle.read(records_member), records_member)
        records = [DestylizationRecord.model_validate(row) for row in record_payloads]
        failures = (
            parse_jsonl_bytes(bundle.read(failures_member), failures_member)
            if failures_member is not None
            else []
        )
        if len(records) != 60:
            raise ValueError(f"archive has {len(records)} success records, expected 60")
        record_ids = [record.source_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("records contain duplicate source IDs")
        if set(record_ids) != set(test_rows):
            missing = sorted(set(test_rows) - set(record_ids))
            extra = sorted(set(record_ids) - set(test_rows))
            raise ValueError(f"test source mismatch; missing={missing[:1]}, extra={extra[:1]}")
        if set(record_ids) & sealed_ids:
            raise ValueError("pilot/calibration source appears in test records")

        output_basenames = [Path(record.output_path).name for record in records]
        if len(output_basenames) != len(set(output_basenames)):
            raise ValueError("declared output basenames are not unique")
        image_members = [
            name
            for name in names
            if "/images/" in f"/{name}" and PurePosixPath(name).suffix.lower() == ".png"
        ]
        if len(image_members) != 60:
            raise ValueError(f"archive has {len(image_members)} PNGs under images, expected 60")
        image_by_basename = {PurePosixPath(name).name: name for name in image_members}
        if len(image_by_basename) != 60 or set(image_by_basename) != set(output_basenames):
            raise ValueError("archive images do not exactly match declared output basenames")

        integrity_signatures = set()
        for record in records:
            frozen = test_rows[record.source_id]
            if record.id != frozen.id or record.style_category != frozen.style_category:
                raise ValueError(f"manifest metadata mismatch for {record.source_id}")
            if not Path(record.input_path).as_posix().endswith(frozen.asset_path.as_posix()):
                raise ValueError(f"input asset path mismatch for {record.source_id}")
            if record.backend != BACKEND or record.seed != 42:
                raise ValueError(f"backend or seed mismatch for {record.source_id}")
            if record.prompt != select_prompt(record.style_category, prompts, adaptive=True):
                raise ValueError(f"prompt mismatch for {record.source_id}")
            required_extra = {
                "official_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
                "source_revision": "master",
                "dtype": "bfloat16",
                "batch_size": 1,
                "height": 1024,
                "width": 1024,
                "max_area": 1024 * 1024,
                "output_height": 1024,
                "output_width": 1024,
                "guidance_scale": 2.5,
                "num_inference_steps": 28,
                "offload": "enable_model_cpu_offload",
                "local_files_only": True,
            }
            mismatches = {
                key: (record.extra.get(key), value)
                for key, value in required_extra.items()
                if record.extra.get(key) != value
            }
            if mismatches:
                raise ValueError(f"frozen setting mismatch for {record.source_id}: {mismatches}")
            signature_fields = (
                "transport_source",
                "transport_model_id",
                "resolved_model_path",
                "download_manifest_sha256",
                "hash_manifest_sha256",
                "package_versions",
            )
            signature = tuple(
                json.dumps(record.extra.get(field), sort_keys=True)
                for field in signature_fields
            )
            if any(
                record.extra.get(field) is None or record.extra.get(field) == ""
                for field in signature_fields
            ):
                raise ValueError(f"missing acquisition/runtime setting for {record.source_id}")
            integrity_signatures.add(signature)
            member = image_by_basename[Path(record.output_path).name]
            with Image.open(io.BytesIO(bundle.read(member))) as image:
                image.load()
                if image.mode != "RGB" or image.size != (1024, 1024):
                    raise ValueError(
                        f"invalid image for {record.source_id}: "
                        f"mode={image.mode}, size={image.size}"
                    )
        if len(integrity_signatures) != 1:
            raise ValueError("acquisition or runtime settings differ across success records")

    failure_ids = [str(row.get("source_id", "")) for row in failures]
    unknown_failures = sorted(set(failure_ids) - set(test_rows))
    if unknown_failures:
        raise ValueError(f"failure record contains non-test source: {unknown_failures[0]}")
    for index, failure in enumerate(failures, start=1):
        if (
            failure.get("backend") != BACKEND
            or failure.get("seed") != 42
            or not failure.get("failure_stage")
            or not failure.get("exception_type")
            or "message" not in failure
        ):
            raise ValueError(f"failure record {index} lacks frozen settings or explicit status")
    report = {
        "schema": "face-destyle-formal-v1-flux-test-archive-validation/v1",
        "archive": str(archive.resolve()),
        "archive_sha256": actual_digest,
        "zip_crc": "passed",
        "success_record_count": len(records),
        "unique_test_source_count": len(set(record_ids)),
        "image_count": len(image_members),
        "image_mode": "RGB",
        "image_size": [1024, 1024],
        "failure_record_count": len(failures),
        "failure_source_count": len(set(failure_ids)),
        "settings": {
            "backend": BACKEND,
            "seed": 42,
            "steps": 28,
            "guidance": 2.5,
            "dtype": "bfloat16",
            "offload": "enable_model_cpu_offload",
        },
        "note": "Machine integrity only; no image was displayed and no quality claim is made.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256-sidecar", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/formal-v1/inputs.jsonl")
    )
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_archive(
            args.archive, args.sha256_sidecar, args.manifest, args.styles_config
        )
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    if args.report.exists():
        parser.error(f"refusing to overwrite report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Validated sealed FLUX test archive: 60 records, 60 RGB 1024 images; "
        f"failures={report['failure_record_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
