import importlib.util
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_origami_lora_run.py"
SPEC = importlib.util.spec_from_file_location("verify_origami_lora_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_fake_safetensors(path: Path, *, payload_size: int = 16) -> None:
    header = json.dumps(
        {
            "adapter.weight": {
                "dtype": "F16",
                "shape": [2, 4],
                "data_offsets": [0, payload_size],
            }
        },
        separators=(",", ":"),
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload_size)


def test_safetensors_inspector_accepts_complete_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "MIN_LORA_BYTES", 1)
    path = tmp_path / "adapter.safetensors"
    write_fake_safetensors(path)

    fingerprint, digest = MODULE.inspect_safetensors(path)

    assert fingerprint == (("adapter.weight", "F16", (2, 4), 16),)
    assert len(digest) == 64


def test_safetensors_inspector_rejects_truncated_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "MIN_LORA_BYTES", 1)
    path = tmp_path / "adapter.safetensors"
    write_fake_safetensors(path)
    path.write_bytes(path.read_bytes()[:-1])

    try:
        MODULE.inspect_safetensors(path)
    except ValueError as exc:
        assert "truncated or unexplained payload" in str(exc)
    else:
        raise AssertionError("truncated safetensors payload was accepted")
