#!/usr/bin/env python3
"""Run a configured destylization backend over JSONL metadata."""

import argparse
from pathlib import Path

from tqdm import tqdm

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.pipelines import CopyBackend, DiffusersBackend
from face_destyle.schemas import ImageRecord
from face_destyle.utils.io import load_yaml
from face_destyle.utils.reproducibility import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--backend", choices=("copy", "diffusers"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    backend_name = args.backend or str(config.get("backend", "copy"))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    seed_everything(seed)
    backend = CopyBackend() if backend_name == "copy" else DiffusersBackend()
    inputs = read_jsonl(args.metadata, ImageRecord, check_paths=True, path_fields=("image_path",))
    outputs = [backend.run(item, args.output_dir, seed=seed) for item in tqdm(inputs)]
    write_jsonl(outputs, args.records_output)
    print(f"Wrote {len(outputs)} {backend_name} records to {args.records_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
