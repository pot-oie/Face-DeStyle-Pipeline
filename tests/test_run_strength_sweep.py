import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_strength_sweep.py"


def test_duplicate_strengths_are_rejected_before_gpu_setup(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(tmp_path / "missing.jsonl"),
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--strength",
            "0.5",
            "0.5",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--strength values must be unique" in result.stderr
