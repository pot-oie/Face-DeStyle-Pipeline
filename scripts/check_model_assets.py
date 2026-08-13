#!/usr/bin/env python3
"""Check configured model files without importing GPU libraries or downloading anything."""

import argparse
from pathlib import Path

from face_destyle.models import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--root", type=Path, help="Override FACE_DESTYLE_ROOT for local assets.")
    parser.add_argument(
        "--asset",
        action="append",
        help="Check only this asset; repeat for multiple assets.",
    )
    args = parser.parse_args()

    registry = ModelRegistry.from_yaml(args.config)
    names = args.asset or sorted(registry.assets)
    failed = 0
    for name in names:
        check = registry.check(name, root=args.root)
        status = "OK" if check.available else "MISSING"
        print(f"{status:7} {name:28} {check.location or '-'}")
        if check.reason:
            print(f"        reason: {check.reason}")
        for missing in check.missing_files:
            print(f"        missing: {missing}")
        failed += not check.available
    print(f"Checked {len(names)} assets; unavailable={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
