import unittest

import biotite.structure as struc
import numpy as np

from synth_pdb.generator import PeptideGenerator, PeptideResult


class TestPeptideGeneratorEnsemble(unittest.TestCase):
    def test_generate_ensemble_stack(self) -> None:
        """Test generating an ensemble as an AtomArrayStack."""
        seq = "ALA-GLY-SER"
        gen = PeptideGenerator(seq)
        n_models = 5
        ensemble = gen.generate_ensemble(n_models=n_models, as_stack=True)

        self.assertIsInstance(ensemble, struc.AtomArrayStack)
        self.assertEqual(ensemble.stack_depth(), n_models)
        # 3 residues, ALA (10), GLY (7), SER (11) = 28 atoms with hydrogens (minus caps)
        self.assertEqual(ensemble.array_length(), 28)

    def test_generate_ensemble_list(self) -> None:
        """Test generating an ensemble as a list of PeptideResult objects."""
        seq = "ALA-GLY-SER"
        gen = PeptideGenerator(seq)
        n_models = 3
        ensemble = gen.generate_ensemble(n_models=n_models, as_stack=False)

        self.assertIsInstance(ensemble, list)
        self.assertEqual(len(ensemble), n_models)
        for res in ensemble:
            self.assertIsInstance(res, PeptideResult)
            self.assertEqual(res.structure.array_length(), 28)

    def test_generate_ensemble_with_drift(self) -> None:
        """Test ensemble generation with torsion drift."""
        seq = "A-A-A-A-A"
        gen = PeptideGenerator(seq)
        n_models = 10
        # Drift should result in structures being different
        ensemble = gen.generate_ensemble(n_models=n_models, drift=10.0)

        # Check that coordinates are not all identical
        coords = ensemble.coord
        self.assertFalse(np.allclose(coords[0], coords[1]))

    def test_generate_ensemble_multichain(self) -> None:
        """Test ensemble generation for multi-chain sequences."""
        seq = "ALA-GLY:SER-THR"
        gen = PeptideGenerator(seq)
        ensemble = gen.generate_ensemble(n_models=2, as_stack=True)
        self.assertEqual(ensemble.stack_depth(), 2)
        # Verify chains are present
        self.assertTrue("A" in ensemble.chain_id)
        self.assertTrue("B" in ensemble.chain_id)


if __name__ == "__main__":
    unittest.main()
