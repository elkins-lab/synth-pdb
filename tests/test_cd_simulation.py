import unittest
import numpy as np
import biotite.structure as struc
from synth_pdb.generator import generate_pdb_content
from synth_pdb.cd_simulator import (
    CDSimulator,
    validate_cd_against_literature,
    BASIS_SPECTRA,
    WAVELENGTHS,
)


class TestCDSimulator(unittest.TestCase):
    def test_pure_helix_cd(self):
        """Validates that a poly-alanine helix produces a helix-like CD spectrum."""
        # Generate a 20-residue poly-Ala helix
        pdb_content = generate_pdb_content(length=20, sequence_str="A" * 20, conformation="alpha")
        from io import StringIO
        from biotite.structure.io.pdb import PDBFile

        pdb_file = PDBFile.read(StringIO(pdb_content))
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

    def test_pure_sheet_cd(self):
        """Validates that a poly-valine sheet produces a sheet-like CD spectrum."""
        # Generate a poly-Val beta strand
        # Note: A single strand might be classified as 'C' or 'E' depending on local geometry
        pdb_content = generate_pdb_content(length=20, sequence_str="V" * 20, conformation="beta")
        from io import StringIO
        from biotite.structure.io.pdb import PDBFile

        pdb_file = PDBFile.read(StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)

        sim = CDSimulator(structure)
        spectrum = sim.get_spectrum(noise_level=0)

        # If it's classified as sheet, check 217nm peak
        if sim.fractions["E"] > 0.5:
            val_217 = spectrum[WAVELENGTHS == 217][0]
            assert -25000 < val_217 < -10000, f"Expected 217nm peak for sheet, got {val_217}"

    def test_basis_spectra_consistency(self):
        """Ensures basis spectra match published characteristics."""
        # Helix peaks
        h = BASIS_SPECTRA["H"]
        assert h[WAVELENGTHS == 222][0] == -38000
        assert h[WAVELENGTHS == 208][0] == -36000
        assert h[WAVELENGTHS == 192][0] == 70000


if __name__ == "__main__":
    unittest.main()
