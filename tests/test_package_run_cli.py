import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_run.py"


def run_package(run_dir, archive, outputs_root, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--archive",
            str(archive),
            "--outputs-root",
            str(outputs_root),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_run_writes_verified_relative_zip_and_checksum(tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-001"
    (run_dir / "images").mkdir(parents=True)
    (run_dir / "images/result.png").write_bytes(b"png-data")
    (run_dir / "records.jsonl").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "exports/run-001.zip"

    result = run_package(run_dir, archive, outputs)

    assert result.returncode == 0, result.stderr
    assert run_dir.is_dir()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert bundle.namelist() == ["run-001/images/result.png", "run-001/records.jsonl"]
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_name("run-001.zip.sha256").read_text(encoding="utf-8") == (
        f"{checksum}  run-001.zip\n"
    )


def test_package_run_cleanup_removes_only_selected_run(tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "selected"
    other_run = outputs / "keep"
    run_dir.mkdir(parents=True)
    other_run.mkdir()
    (run_dir / "result.txt").write_text("result", encoding="utf-8")
    (other_run / "result.txt").write_text("keep", encoding="utf-8")
    archive = tmp_path / "exports/selected.zip"

    result = run_package(run_dir, archive, outputs, "--cleanup")

    assert result.returncode == 0, result.stderr
    assert not run_dir.exists()
    assert other_run.is_dir()
    assert archive.is_file()
    assert archive.with_name("selected.zip.sha256").is_file()


def test_package_run_refuses_outputs_root_as_cleanup_target(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "result.txt").write_text("result", encoding="utf-8")

    result = run_package(outputs, tmp_path / "archive.zip", outputs, "--cleanup")

    assert result.returncode == 2
    assert "entire outputs root" in result.stderr


def test_package_run_rejects_symlinks(tmp_path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    (run_dir / "linked.txt").symlink_to(outside)

    result = run_package(run_dir, tmp_path / "archive.zip", outputs)

    assert result.returncode == 2
    assert "contains symlinks" in result.stderr
