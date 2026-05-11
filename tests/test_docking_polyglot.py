import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from synth_pdb.main import main


class TestDockingPolyglot:
    """Tests the docking mode CLI integration with multiple output formats."""

    @pytest.mark.parametrize("fmt", ["pdb", "cif", "bcif"])
    def test_docking_mode_with_formats(self, tmp_path: Path, fmt: str) -> None:
        """Test that --mode docking produces a PQR file for all output formats.

        This covers the 'Generating new structure first' path.
        """
        output_base = tmp_path / f"test_docking_{fmt}"
        output_file = output_base.with_suffix(f".{fmt}")
        pqr_file = output_base.with_suffix(".pqr")

        test_args = [
            "synth_pdb",
            "--mode",
            "docking",
            "--sequence",
            "ALA-ALA",
            "--format",
            fmt,
            "--output",
            str(output_file),
        ]

        with patch("sys.argv", test_args):
            # We don't need to mock anything, let it run a small generation
            main()

        assert output_file.exists(), f"Primary output {fmt} file should exist"
        assert pqr_file.exists(), f"PQR file should exist for format {fmt}"

        pqr_content = pqr_file.read_text()
        assert "ATOM" in pqr_content
        # Check for charge/radius columns (PQR specific)
        for line in pqr_content.splitlines():
            if line.startswith("ATOM"):
                # Column 55-62 (Charge), 63-70 (Radius)
                charge = line[54:62].strip()
                radius = line[62:70].strip()
                assert charge, f"Missing charge in PQR for format {fmt}"
                assert radius, f"Missing radius in PQR for format {fmt}"

    def test_docking_mode_input_pdb(self, tmp_path: Path) -> None:
        """Test --mode docking with an existing --input-pdb."""
        from synth_pdb.generator import generate_pdb_content

        input_pdb = tmp_path / "existing.pdb"
        # Generate a real valid PDB
        pdb_content = generate_pdb_content(length=3, sequence_str="ALA-ALA-ALA")
        input_pdb.write_text(pdb_content)

        output_pqr = tmp_path / "converted.pqr"

        test_args = [
            "synth_pdb",
            "--mode",
            "docking",
            "--input-pdb",
            str(input_pdb),
            "--output",
            str(output_pqr),
        ]

        with patch("sys.argv", test_args):
            main()

        assert output_pqr.exists()
        assert "ALA" in output_pqr.read_text()
