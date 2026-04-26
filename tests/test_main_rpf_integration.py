import os
import subprocess
import sys
from typing import Any

# Save the absolute path to the project root at the module level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_main_rpf_validation_cli(tmp_path: Any, monkeypatch: Any) -> None:
    """
    TDD Test: CLI Integration of RPF Score.
    """
    monkeypatch.chdir(tmp_path)

    # 1. Create a mock restraint file
    restraints_content = "1 HN 5 HN 5.0\n1 HA 6 HN 3.5\n"
    restraints_path = tmp_path / "mock.restraints"
    restraints_path.write_text(restraints_content)

    # 2. Run synth-pdb with --restraints
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "synth_pdb.main",
        "--sequence",
        "AKAAKAAK",
        "--minimize",  # Ensure we have Hydrogens
        "--validate",
        "--restraints",
        str(restraints_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    # 3. Verify Output
    assert (
        "--- NMR RPF Validation Report ---" in result.stdout
        or "--- NMR RPF Validation Report ---" in result.stderr
    )
    assert "Recall:" in result.stdout or "Recall:" in result.stderr
    assert "Precision:" in result.stdout or "Precision:" in result.stderr
    assert "F-measure:" in result.stdout or "F-measure:" in result.stderr


def test_main_rpf_validation_nef_cli(tmp_path: Any, monkeypatch: Any) -> None:
    """
    Ensures that the CLI correctly validates against a standard NEF file.
    """
    monkeypatch.chdir(tmp_path)

    # 1. Generate a PDB and its own NEF restraints
    truth_pdb = "truth.pdb"
    truth_nef = "truth.nef"
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "synth_pdb.main",
            "--sequence",
            "ALALALA",
            "--minimize",
            "--gen-nef",
            "--output",
            truth_pdb,
            "--nef-output",
            truth_nef,
        ],
        env=env,
        check=True,
    )

    # 2. Now run validation against this NEF
    cmd = [
        sys.executable,
        "-m",
        "synth_pdb.main",
        "--sequence",
        "ALALALA",
        "--minimize",
        "--restraints",
        truth_nef,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # 3. Verify Output
    assert (
        "--- NMR RPF Validation Report ---" in result.stdout
        or "--- NMR RPF Validation Report ---" in result.stderr
    )
    assert truth_nef in (result.stdout + result.stderr)
    assert "F-measure:" in (result.stdout + result.stderr)
