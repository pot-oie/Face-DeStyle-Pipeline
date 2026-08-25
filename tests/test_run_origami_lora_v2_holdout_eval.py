import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_origami_lora_v2_holdout_eval.sh"


def test_origami_v2_holdout_script_has_valid_bash_syntax_and_help() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "fixed six-source Origami holdout" in help_result.stdout


def test_origami_v2_holdout_matrix_and_settings_are_frozen() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_method base" in text
    assert 'run_method v1-checkpoint-100' in text
    assert "for checkpoint in 50 100 150 200" in text
    assert '--seed 42' in text
    assert '--num-inference-steps 28' in text
    assert '--guidance-scale 2.5' in text
    assert '--lora-scale 1.0' in text
    assert "--resume" in text
    assert "tmux" not in text
    assert "checkpoint-300" not in text
