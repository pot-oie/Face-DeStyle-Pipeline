import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

from PIL import Image

from face_destyle.data.metadata import write_jsonl
from face_destyle.schemas import DatasetManifestRecord, DestylizationRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_heldout_blind_review.py"
SPEC = importlib.util.spec_from_file_location("build_heldout_blind_review", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_five_method_matrix_and_stratified_repeat(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    manifest_rows = []
    records_by_method = [[] for _ in range(5)]
    for style_index, style in enumerate(MODULE.STYLES):
        for index in range(15):
            source_id = f"{style}-{index:02d}"
            source = data_root / "raw" / style / f"{source_id}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (style_index * 30, index, 1)).save(source)
            manifest_rows.append(
                DatasetManifestRecord(
                    id=source_id,
                    source_id=source_id,
                    source_group_id=f"group-{source_id}",
                    asset_path=source.relative_to(data_root),
                    style_category=style,
                    split="test",
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                )
            )
            for method_index, rows in enumerate(records_by_method):
                candidate = output_root / f"method-{method_index}" / f"{source_id}.png"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (method_index, style_index, index)).save(candidate)
                rows.append(
                    DestylizationRecord(
                        id=source_id,
                        source_id=source_id,
                        input_path=source,
                        output_path=candidate,
                        style_category=style,
                        backend=f"backend-{method_index}",
                        seed=42,
                    )
                )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest_rows, manifest)
    method_args = []
    for method_index, rows in enumerate(records_by_method):
        path = tmp_path / f"method-{method_index}.jsonl"
        write_jsonl(rows, path)
        method_args.append((f"method-{method_index}", path))

    candidates, methods = MODULE.collect_candidates(manifest, data_root, method_args)
    repeat = MODULE.select_repeat(candidates)

    assert len(methods) == 5
    assert len(candidates) == 300
    assert len({row["canonical_id"] for row in candidates}) == 300
    assert len(repeat) == 60
    assert len({row["canonical_id"] for row in repeat}) == 60
    assert set(Counter((row["method"], row["style_category"]) for row in repeat).values()) == {3}
    assert [row["canonical_id"] for row in repeat] == [
        row["canonical_id"] for row in MODULE.select_repeat(candidates)
    ]
