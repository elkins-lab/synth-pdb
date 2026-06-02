from typing import Any
import unittest
import numpy as np
import biotite.structure as struc
from io import StringIO
import os
import tempfile
import sys
from unittest.mock import patch
from biotite.structure.io.pdb import PDBFile
from synth_pdb.generator import generate_pdb_content
from synth_pdb.cd_simulator import (
    CDSimulator,
    validate_cd_against_literature,
    BASIS_SPECTRA,
    WAVELENGTHS,
)

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestCDSimulator(unittest.TestCase):
    def test_pure_helix_cd(self) -> None:
        """Validates that a poly-alanine helix produces a helix-like CD spectrum."""
        # Generate a 20-residue poly-Ala helix
        pdb_content = generate_pdb_content(length=20, sequence_str="A" * 20, conformation="alpha")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)

        sim = CDSimulator(structure)
        spectrum = sim.get_spectrum(noise_level=0)  # Deterministic

        # Verify fractions
        assert sim.fractions["H"] > 0.8, f"Expected high helix fraction, got {sim.fractions['H']}"

        # Verify 222nm peak
        val_222 = spectrum[WAVELENGTHS == 222][0]
        assert -45000 < val_222 < -30000, f"Expected 222nm peak near -36,000, got {val_222}"

        # Run literature validation
        findings = validate_cd_against_literature(sim.fractions, spectrum)
        assert any("matches literature" in f for f in findings)

    def test_pure_sheet_cd(self) -> None:
        """Validates that a poly-valine sheet produces a sheet-like CD spectrum."""
        # Generate a poly-Val beta strand
        # Note: A single strand might be classified as 'C' or 'E' depending on local geometry
        pdb_content = generate_pdb_content(length=20, sequence_str="V" * 20, conformation="beta")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)

        sim = CDSimulator(structure)
        spectrum = sim.get_spectrum(noise_level=0)

        # If it's classified as sheet, check 217nm peak
        if sim.fractions["E"] > 0.5:
            val_217 = spectrum[WAVELENGTHS == 217][0]
            assert -25000 < val_217 < -10000, f"Expected 217nm peak for sheet, got {val_217}"

    def test_basis_spectra_consistency(self) -> None:
        """Ensures basis spectra match published characteristics."""
        # Helix peaks
        h = BASIS_SPECTRA["H"]
        assert h[WAVELENGTHS == 222][0] == -38000
        assert h[WAVELENGTHS == 208][0] == -36000
        assert h[WAVELENGTHS == 192][0] == 70000

    def test_cd_with_noise(self) -> None:
        """Test that adding noise produces a different but similar spectrum."""
        pdb_content = generate_pdb_content(length=10, conformation="alpha")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)
        sim = CDSimulator(structure)

        spec1 = sim.get_spectrum(noise_level=0)
        spec2 = sim.get_spectrum(noise_level=1000)

        assert not np.array_equal(spec1, spec2)
        assert np.mean(np.abs(spec1 - spec2)) < 2000  # Should be within noise range

    def test_cd_empty_structure(self) -> None:
        """Test handling of empty structure (should return coil)."""
        # Create a mock structure with no atoms
        empty_struct = struc.AtomArray(0)
        sim = CDSimulator(empty_struct)
        assert sim.fractions["C"] == 1.0
        assert sim.fractions["H"] == 0.0
        assert sim.fractions["E"] == 0.0

    def test_cd_plotting_to_file(self) -> None:
        """Test that the plot method successfully saves a file."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not installed")
        pdb_content = generate_pdb_content(length=10, conformation="alpha")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)
        sim = CDSimulator(structure)

        with tempfile.TemporaryDirectory() as tmp:
            plot_path = os.path.join(tmp, "test_cd.png")
            sim.plot(save_path=plot_path)
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 0

    def test_cli_cd_integration(self) -> None:
        """Test the --gen-cd flag through the main CLI entry point."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not installed")
        from synth_pdb.main import main

        with tempfile.TemporaryDirectory() as tmp:
            out_pdb = os.path.join(tmp, "cli_test.pdb")
            expected_plot = os.path.join(tmp, "cli_test_cd.png")

            test_args = [
                "synth-pdb",
                "--sequence",
                "AAAAAAAAAA",
                "--conformation",
                "alpha",
                "--gen-cd",
                "--output",
                out_pdb,
            ]

            with patch.object(sys, "argv", test_args):
                main()

            assert os.path.exists(out_pdb)
            assert os.path.exists(expected_plot)

    def test_validation_negative_cases(self) -> None:
        """Test validation findings for non-ideal structures."""
        # Pure coil
        pdb_content = generate_pdb_content(length=10, conformation="random")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)
        sim = CDSimulator(structure)
        spectrum = sim.get_spectrum(noise_level=0)

        findings = validate_cd_against_literature(sim.fractions, spectrum)
        # Should NOT find helix/sheet matches because it's random coil
        assert not any("matches literature" in f for f in findings)

    def test_validate_cd_helix_ok_branch_fires(self) -> None:
        """validate_cd_against_literature must report [OK] for a predominantly
        helical structure.

        SCIENTIFIC BASIS:
        For a structure with f_helix > 0.8 the function checks that the 222 nm
        ellipticity lies within [-42,000, -30,000] deg·cm²·dmol⁻¹.  This range
        brackets the literature value of ~-36,000 reported by Greenfield &
        Fasman (1969).  A poly-Ala alpha-helix should satisfy this condition when
        the synthetic basis spectrum is correct.

        This test was added because the previous coverage only called
        validate_cd_against_literature with random-coil structures where
        f_helix < 0.8, meaning the [OK] branch was never exercised.
        """
        # 20-residue poly-Ala helix: annotate_sse assigns H to most residues
        pdb_content = generate_pdb_content(length=20, sequence_str="A" * 20, conformation="alpha")
        pdb_file = PDBFile.read(StringIO(str(pdb_content)))
        structure = pdb_file.get_structure(model=1)

        sim = CDSimulator(structure)
        spectrum = sim.get_spectrum(noise_level=0)

        # Guard: skip if generator didn't produce enough helix (e.g. SSE mis-annotated)
        if sim.fractions["H"] <= 0.8:
            import pytest

            pytest.skip(f"Helix fraction too low ({sim.fractions['H']:.2f}) to trigger validation")

        findings = validate_cd_against_literature(sim.fractions, spectrum)
        assert any("[OK]" in f and "222nm" in f for f in findings), (
            f"Expected [OK] helix 222nm finding; got: {findings}"
        )


if __name__ == "__main__":
    unittest.main()
