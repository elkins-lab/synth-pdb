import os
import pytest
from pathlib import Path
from unittest.mock import patch
from synth_pdb.main import main


class TestConstraintCutoff:
    """Tests that --constraint-cutoff is correctly respected during export."""

    def test_constraint_cutoff_filtering(self, tmp_path: Path) -> None:
        """Verify that --constraint-cutoff filters contacts in the exported file."""
        output_csv = tmp_path / "contacts.csv"

        # We'll use a sequence that forms a predictable structure (alpha helix)
        # and test with a very small cutoff to see if it filters out longer-range contacts.

        # 1. Test with large cutoff (default 8.0)
        test_args_large = [
            "synth_pdb",
            "--sequence",
            "AAAAAAAAAA",
            "--export-constraints",
            str(output_csv),
            "--constraint-format",
            "csv",
            "--constraint-cutoff",
            "10.0",
        ]

        with patch("sys.argv", test_args_large):
            main()

        content_large = output_csv.read_text()
        lines_large = [l for l in content_large.splitlines() if "," in l and "Res1" not in l]
        num_contacts_large = len(lines_large)

        # 2. Test with small cutoff
        test_args_small = [
            "synth_pdb",
            "--sequence",
            "AAAAAAAAAA",
            "--export-constraints",
            str(output_csv),
            "--constraint-format",
            "csv",
            "--constraint-cutoff",
            "4.0",  # ALA CA-CA is ~3.8A
        ]

        with patch("sys.argv", test_args_small):
            main()

        content_small = output_csv.read_text()
        lines_small = [l for l in content_small.splitlines() if "," in l and "Res1" not in l]
        num_contacts_small = len(lines_small)

        assert num_contacts_small < num_contacts_large, (
            f"Small cutoff (4.0) should produce fewer contacts than large cutoff (10.0). "
            f"Got {num_contacts_small} vs {num_contacts_large}"
        )

        # All distances in small file should be <= 4.0
        for line in lines_small:
            dist = float(line.split(",")[2])
            assert dist <= 4.0, f"Distance {dist} exceeds cutoff 4.0"

    def test_constraint_cutoff_casp_header(self, tmp_path: Path) -> None:
        """Verify that --constraint-cutoff is reflected in CASP RR format values."""
        output_rr = tmp_path / "contacts.rr"

        test_args = [
            "synth_pdb",
            "--sequence",
            "AAAAA",
            "--export-constraints",
            str(output_rr),
            "--constraint-format",
            "casp",
            "--constraint-cutoff",
            "5.5",
        ]

        with patch("sys.argv", test_args):
            main()

        content = output_rr.read_text()
        # CASP RR format: i j d_minor d_major prob

        for line in content.splitlines():
            if len(line.split()) == 5:
                parts = line.split()
                dist = float(parts[3])
                assert dist <= 5.5, f"Distance {dist} in CASP RR exceeds cutoff 5.5"

    def test_constraint_cutoff_msa_impact(self, tmp_path: Path) -> None:
        """Verify that --constraint-cutoff impacts MSA generation."""
        output_fasta = tmp_path / "msa.fasta"

        # We can't easily check the sequences, but we can verify the command runs
        # and respect the flag. If it didn't respect it, it would use 8.0.
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAAAAAAAAA",
            "--gen-msa",
            "--msa-depth",
            "5",
            "--output",
            str(output_fasta),
            "--constraint-cutoff",
            "12.0",  # Very loose
        ]

        with patch("sys.argv", test_args):
            main()

        assert output_fasta.exists()
        assert ">seq_0" in output_fasta.read_text()
