import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Save the absolute path to the project root at the module level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_main_rdc_validation_cli(tmp_path: Path, monkeypatch: Any) -> None:
    """
    TDD Test: CLI Integration of RDC Q-factor.
    """
    monkeypatch.chdir(tmp_path)

    # 1. Create a mock RDC file
    # Format: res_1 atom_1 res_2 atom_2 value
    rdc_content = "1 N 1 HN 10.0\n2 N 2 HN -5.0\n"
    rdc_path = tmp_path / "mock.rdcs"
    rdc_path.write_text(rdc_content)

    # 2. Run synth-pdb with --rdc-restraints
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "synth_pdb.main",
        "--sequence",
        "ALALALA",
        "--minimize",
        "--rdc-restraints",
        str(rdc_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # 3. Verify Output
    assert (
        "--- NMR RDC Validation Report ---" in result.stdout
        or "--- NMR RDC Validation Report ---" in result.stderr
    )
    assert "Q-factor:" in result.stdout or "Q-factor:" in result.stderr
