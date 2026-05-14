import os
import unittest
import io
import numpy as np
import biotite.structure as struc
from synth_pdb.generator import generate_pdb_content, PeptideResult


class TestOutputFormats(unittest.TestCase):
    def setUp(self):
        self.sequence = "ALA-GLY-SER"

    def test_pdb_format(self):
        """Verify standard PDB output."""
        content = generate_pdb_content(sequence_str=self.sequence, output_format="pdb")
        self.assertIsInstance(content, str)
        self.assertIn("ATOM", content)
        self.assertIn("TER", content)

        # Verify it can be re-parsed
        res = PeptideResult(content, format="pdb")
        self.assertEqual(res.structure.array_length(), 31)

    def test_cif_format(self):
        """Verify mmCIF (PDBx) output."""
        content = generate_pdb_content(sequence_str=self.sequence, output_format="cif")
        self.assertIsInstance(content, str)
        self.assertIn("_atom_site.group_PDB", content)
        self.assertIn("data_structure", content)

        # Verify it can be re-parsed
        res = PeptideResult(content, format="cif")
        self.assertEqual(res.structure.array_length(), 31)
        # Check if B-factors were preserved
        self.assertTrue(np.any(res.structure.b_factor > 0))

    def test_bcif_format(self):
        """Verify BinaryCIF output."""
        content = generate_pdb_content(sequence_str=self.sequence, output_format="bcif")
        self.assertIsInstance(content, bytes)

        # Verify it can be re-parsed
        res = PeptideResult(content, format="bcif")
        self.assertEqual(res.structure.array_length(), 31)
        self.assertTrue(np.any(res.structure.b_factor > 0))

    def test_format_conversion(self):
        """Test the get_content method for cross-format conversion."""
        res = PeptideResult(
            generate_pdb_content(sequence_str=self.sequence, output_format="pdb"), format="pdb"
        )

        cif_content = res.get_content("cif")
        self.assertIsInstance(cif_content, str)
        self.assertIn("_atom_site.group_PDB", cif_content)

        bcif_content = res.get_content("bcif")
        self.assertIsInstance(bcif_content, bytes)

    def test_bcif_after_minimization(self):
        """Verify that BinaryCIF output works correctly after energy minimization."""
        content = generate_pdb_content(
            sequence_str="ALA-GLY", minimize_energy=True, output_format="bcif"
        )
        self.assertIsInstance(content, bytes)

        # Re-parse
        res = PeptideResult(content, format="bcif")
        self.assertTrue(res.structure.array_length() >= 17)  # ALA + GLY + protons
        self.assertTrue(np.any(res.structure.b_factor > 0))

    def test_massive_coordinate_overflow(self):
        """Verify that mmCIF/BCIF handle coordinates > 1000A (PDB overflow)."""
        # Create a tiny AtomArray but with massive coordinates
        array = struc.AtomArray(1)
        array.res_name = np.array(["ALA"])
        array.atom_name = np.array(["CA"])
        array.res_id = np.array([1])
        array.chain_id = np.array(["A"])
        array.coord = np.array([[1234.567, 2345.678, 3456.789]])
        array.hetero = np.array([False])
        array.element = np.array(["C"])

        # Directly use PeptideResult with the array
        res = PeptideResult("DUMMY", format="pdb")
        res._structure = array

        # mmCIF should work and contain the full coordinate
        cif = res.get_content("cif")
        self.assertIn("1234.567", cif)
        self.assertIn("2345.678", cif)
        self.assertIn("3456.789", cif)

        # BinaryCIF should work and be re-parsable
        bcif = res.get_content("bcif")
        res_back = PeptideResult(bcif, format="bcif")
        np.testing.assert_allclose(res_back.structure.coord, array.coord)


if __name__ == "__main__":
    unittest.main()
