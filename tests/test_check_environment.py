import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_environment.py"


def test_environment_check_rejects_invalid_omp_threads():
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "invalid"

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "OMP_NUM_THREADS must be a positive integer" in result.stdout
