import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(args: list[str]) -> tuple[int, str, str]:
    """Helper to run the CLI and return returncode, stdout, stderr."""
    cmd = [sys.executable, "-m", "synth_pdb.main"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


class TestCLICollisions:
    """
    Test suite for CLI argument collisions and validation.
    Ensures the tool fails gracefully with helpful messages.
    """

    def test_invalid_length(self) -> None:
        """Verify that negative or zero length is rejected."""
        # Zero length
        code, _, err = run_cli(["--length", "0"])
        assert code != 0
        assert "Length must be a positive integer" in err

        # Negative length
        code, _, err = run_cli(["--length", "-5"])
        assert code != 0
        assert "Length must be a positive integer" in err

    def test_docking_mode_missing_input(self) -> None:
        """Verify that docking mode fails if --input-pdb is missing."""
        code, _, err = run_cli(["--mode", "docking"])
        assert code != 0
        assert "Docking mode requires --input-pdb" in err

    def test_pymol_mode_missing_inputs(self) -> None:
        """Verify that pymol mode fails if required inputs are missing."""
        code, _, err = run_cli(["--mode", "pymol", "--input-pdb", "test.pdb"])
        assert code != 0
        assert "PyMOL mode requires" in err

    def test_ai_mode_missing_op(self) -> None:
        """Verify that ai mode fails if --ai-op is missing or invalid."""
        code, _, err = run_cli(["--mode", "ai"])
        assert code != 0
        assert "AI mode requires --ai-op" in err

        code, _, err = run_cli(["--mode", "ai", "--ai-op", "junk"])
        assert code != 0
        # argparse will catch invalid choice
        assert "invalid choice" in err

    def test_ai_interpolate_missing_pdbs(self) -> None:
        """Verify that interpolation fails if start/end PDBs are missing."""
        code, _, err = run_cli(["--mode", "ai", "--ai-op", "interpolate"])
        assert code != 0
        assert "Interpolation requires --start-pdb and --end-pdb" in err

    def test_decoys_mode_missing_seq_and_length(self) -> None:
        """Verify that decoys mode fails if neither sequence nor length is provided."""
        code, _, _ = run_cli(["--mode", "decoys", "--length", "0"])
        # This should trigger the length validation
        assert code != 0

    def test_cyclic_vs_capping_priority(self, tmp_path: Path) -> None:
        """
        Verify that --cyclic correctly disables --cap-termini internally.
        We check the REMARK 3 in the output PDB if possible.
        """
        out_file = tmp_path / "cyclic_test.pdb"
        # We use a very short sequence for speed, and no minimization to skip OpenMM dependency in basic CLI test
        # Actually --cyclic implies --minimize. Let's use a mock or skip if no OpenMM.
        from synth_pdb.physics import HAS_OPENMM

        if not HAS_OPENMM:
            pytest.skip("Cyclic CLI test requires OpenMM")

        code, _, _ = run_cli(
            [
                "--sequence",
                "AAA",
                "--cyclic",
                "--cap-termini",  # Contradictory
                "--output",
                str(out_file),
            ]
        )

        assert code == 0
        with open(out_file) as f:
            content = f.read()
            # REMARK 3 records the command line used.
            assert "cyclic" in content
            # If ACE/NME were added, we'd see those residue names
            assert "ACE" not in content
            assert "NME" not in content

    def test_invalid_rmsd_range(self) -> None:
        """Verify that bad RMSD range formats are handled with a warning (not a crash)."""
        # main.py L256 uses try-except to catch bad RMSD
        # It should log a warning and continue.
        code, out, err = run_cli(
            ["--mode", "decoys", "--length", "5", "--rmsd-range", "garbage", "--n-decoys", "1"]
        )
        # Should NOT crash (code 0 or at least not traceback)
        assert "Invalid RMSD range" in err or "Invalid RMSD range" in out

    def test_conflicting_length_and_sequence(self, tmp_path: Path) -> None:
        """Verify that --sequence overrides --length."""
        out_file = tmp_path / "seq_test.pdb"
        code, _, _ = run_cli(["--length", "50", "--sequence", "AAA", "--output", str(out_file)])
        assert code == 0
        with open(out_file) as f:
            content = f.read()
            # Should have 3 residues, not 50
            assert "LENGTH: 3" in content

    def test_log_level_validation(self) -> None:
        """Verify invalid log levels are rejected by argparse."""
        code, _, err = run_cli(["--log-level", "ULTRA_DEBUG"])
        assert code != 0
        assert "invalid choice" in err
