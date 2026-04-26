import os
import pathlib
import subprocess
import sys
from typing import Any

# Save the absolute path to the project root at the module level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_main_shift_validation_cli(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """
    TDD Test: CLI Integration of Chemical Shift Validation.
    """
    monkeypatch.chdir(tmp_path)

    # 1. Create a mock shift file
    # Format: res_id atom_name value
    # ALA 1 HA is typically ~4.3
    shift_content = "1 HA 4.3\n2 HA 4.3\n"
    shift_path = tmp_path / "mock.shifts"
    shift_path.write_text(shift_content)

    # 2. Run synth-pdb with --shift-restraints
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "synth_pdb.main",
        "--sequence",
        "AA",
        "--minimize",
        "--shift-restraints",
        str(shift_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # 3. Verify Output
    assert (
        "--- NMR Chemical Shift Validation Report ---" in result.stdout
        or "--- NMR Chemical Shift Validation Report ---" in result.stderr
    )
    assert "RMSD:" in result.stdout or "RMSD:" in result.stderr
    assert "Correlation:" in result.stdout or "Correlation:" in result.stderr
