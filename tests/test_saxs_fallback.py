import unittest
import sys
from unittest.mock import MagicMock, patch
import biotite.structure as struc
import numpy as np


class TestSaxsFallback(unittest.TestCase):
    def test_saxs_fallback_logic(self) -> None:
        """Test that SAXS module handles missing synth-saxs gracefully."""
        # 1. Mock synth-saxs being missing
        with patch.dict(sys.modules, {"synth_saxs": None}):
            # We need to reload the module to trigger the ImportError handling
            if "synth_pdb.saxs" in sys.modules:
                del sys.modules["synth_pdb.saxs"]

            import synth_pdb.saxs as saxs

            self.assertFalse(saxs.HAS_SYNTH_SAXS)

            # 2. Test calculate_radius_of_gyration (should work via biotite)
            atoms = struc.AtomArray(2)
            atoms.coord = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
            atoms.res_name = np.array(["ALA", "ALA"])
            atoms.element = np.array(["C", "C"])
            rg = saxs.calculate_radius_of_gyration(atoms)
            self.assertGreater(rg, 0)

            # 3. Test that other functions raise ImportError
            with self.assertRaises(ImportError):
                saxs.calculate_saxs_profile(atoms)

            with self.assertRaises(ImportError):
                saxs.SaxsSimulator(atoms)

    def test_saxs_with_package(self) -> None:
        """Test that SAXS module re-exports correctly when synth-saxs is present."""
        # This assumes synth-saxs IS installed in the environment (which it is)
        import synth_pdb.saxs as saxs

        if saxs.HAS_SYNTH_SAXS:
            self.assertTrue(hasattr(saxs, "calculate_saxs_profile"))
            # Just a sanity check that it's a callable
            self.assertTrue(callable(saxs.calculate_saxs_profile))


if __name__ == "__main__":
    unittest.main()
