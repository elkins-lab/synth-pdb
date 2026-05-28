import unittest
import numpy as np
from synth_pdb.generator import _assemble_output
import biotite.structure as struc


class TestAssembly(unittest.TestCase):
    def setUp(self) -> None:
        # Create a small peptide AtomArray
        self.peptide = struc.AtomArray(5)
        self.peptide.res_id = np.array([1, 1, 1, 1, 1])
        self.peptide.res_name = np.array(["ALA"] * 5)
        self.peptide.atom_name = np.array(["N", "CA", "C", "O", "CB"])
        self.peptide.element = np.array(["N", "C", "C", "O", "C"])
        self.peptide.coord = np.zeros((5, 3))
        self.peptide.hetero = np.zeros(5, dtype=bool)
        self.peptide.chain_id = np.array(["A"] * 5)

    def test_assemble_pdb(self) -> None:
        """Test assembly into legacy PDB format."""
        # Provide some dummy atomic content to trigger patching logic
        atomic_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )

        result = _assemble_output(
            self.peptide,
            atomic_and_ter_content=atomic_content,
            sequence_length=1,
            cyclic=False,
            output_format="pdb",
        )
        self.assertIsInstance(result, str)
        self.assertIn("ATOM", result)
        self.assertIn("REMARK", result)

    def test_assemble_cif(self) -> None:
        """Test assembly into mmCIF format."""
        result = _assemble_output(
            self.peptide,
            atomic_and_ter_content=None,
            sequence_length=1,
            cyclic=False,
            output_format="cif",
        )
        self.assertIsInstance(result, str)
        self.assertIn("_atom_site", result)

    def test_assemble_bcif(self) -> None:
        """Test assembly into BinaryCIF format."""
        result = _assemble_output(
            self.peptide,
            atomic_and_ter_content=None,
            sequence_length=1,
            cyclic=False,
            output_format="bcif",
        )
        self.assertIsInstance(result, bytes)
        # BCIF starts with a specific magic header or structure
        self.assertTrue(len(result) > 0)

    def test_assemble_pdb_with_ptm(self) -> None:
        """Test assembly with PTMs which should be marked as ATOM."""
        self.peptide.res_name[:] = "SEP"
        result = _assemble_output(
            self.peptide,
            atomic_and_ter_content=None,
            sequence_length=1,
            cyclic=False,
            output_format="pdb",
        )
        # Even though they might be in hetero lists elsewhere, SEP should be ATOM
        self.assertIn("ATOM", result)


if __name__ == "__main__":
    unittest.main()
