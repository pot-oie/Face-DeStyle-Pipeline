#!/usr/bin/env python3
"""Structurally verify an Origami LoRA run and its expected checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

MIN_LORA_BYTES = 100_000_000


def inspect_safetensors(path: Path) -> tuple[tuple, str]:
    """Validate the complete safetensors byte layout and return a shape fingerprint."""
    size = path.stat().st_size
    if size < MIN_LORA_BYTES:
        raise ValueError(f"LoRA file is unexpectedly small ({size} bytes): {path}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        (header_length,) = struct.unpack("<Q", prefix)
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header JSON: {path}") from exc
    tensors = []
    intervals = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        try:
            start, end = value["data_offsets"]
            dtype = value["dtype"]
            shape = tuple(value["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid tensor header for {key}: {path}") from exc
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"invalid tensor offsets for {key}: {path}")
        intervals.append((start, end, key))
        tensors.append((key, dtype, shape, end - start))
    if not tensors:
        raise ValueError(f"safetensors file contains no tensors: {path}")
    intervals.sort()
    expected_start = 0
    for start, end, key in intervals:
        if start != expected_start:
            raise ValueError(f"non-contiguous tensor payload before {key}: {path}")
        expected_start = end
    payload_bytes = size - 8 - header_length
    if expected_start != payload_bytes:
        raise ValueError(
            f"truncated or unexplained payload ({expected_start} != {payload_bytes}): {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return tuple(sorted(tensors)), digest.hexdigest()


def verify_run(
    output_dir: Path,
    checkpoints: tuple[int, ...],
    *,
    require_final: bool = True,
) -> list[tuple[Path, int, str]]:
    paths = [
        output_dir / f"checkpoint-{checkpoint}" / "pytorch_lora_weights.safetensors"
        for checkpoint in checkpoints
    ]
    if require_final:
        paths.append(output_dir / "pytorch_lora_weights.safetensors")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing expected LoRA weights: " + ", ".join(missing))
    reference = None
    records = []
    for path in paths:
        fingerprint, digest = inspect_safetensors(path)
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError(f"LoRA tensor keys/shapes differ: {path}")
        records.append((path, path.stat().st_size, digest))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=int,
        action="append",
        dest="checkpoints",
        required=True,
    )
    parser.add_argument("--allow-missing-final", action="store_true")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    try:
        records = verify_run(
            args.output_dir,
            tuple(args.checkpoints),
            require_final=not args.allow_missing_final,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for path, size, digest in records:
        print(f"LORA_OK={path} BYTES={size} SHA256={digest}")
    if args.log:
        if not args.log.is_file():
            parser.error(f"training log is missing: {args.log}")
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
        reached_200 = bool(re.search(r"\b200/200\b", log_text))
        print(f"LOG_REACHED_200={'YES' if reached_200 else 'NO'}")
        if not reached_200:
            parser.error("training log does not contain 200/200")
    print("LORA_RUN_COMPLETE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
