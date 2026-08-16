from pathlib import Path

from PIL import Image

from face_destyle.data.metadata import write_jsonl
from face_destyle.schemas import DatasetManifestRecord


def test_build_blind_review_cli(tmp_path: Path, monkeypatch) -> None:
    import runpy
    import sys

    data_root = tmp_path / "data"
    source_path = data_root / "raw" / "comic" / "one.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 48), "red").save(source_path)
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    Image.new("RGB", (48, 32), "blue").save(method_dir / "one.png")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        [
            DatasetManifestRecord(
                id="one",
                source_id="one",
                source_group_id="group-one",
                asset_path=Path("raw/comic/one.png"),
                style_category="comic",
                split="pilot",
                sha256=(
                    "7a5b756c49720c5a3842039d9bc2ce25fbf95312ef4a5d7236e2439f87d9c328"
                ),
            )
        ],
        manifest,
    )
    # Use the real file checksum expected by the manifest loader.
    import hashlib
    import json

    payload = json.loads(manifest.read_text())
    payload["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n")
    output = tmp_path / "review"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_blind_review.py",
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--method",
            f"method-a={method_dir}",
            "--output-dir",
            str(output),
        ],
    )
    try:
        runpy.run_path("scripts/build_blind_review.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    assert len(list((output / "reviewer/round-a/pairs").glob("*.jpg"))) == 1
    assert len(list((output / "reviewer/round-b/pairs").glob("*.jpg"))) == 1
    assert (output / "private/private_key.jsonl").is_file()
