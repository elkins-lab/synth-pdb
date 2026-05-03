import unittest
import numpy as np
from synth_pdb.validator import PDBValidator
from synth_pdb.generator import create_atom_line


class TestStericClashRobustness(unittest.TestCase):
    def test_apply_steric_clash_tweak_exact_superposition(self):
        """Test that exactly superimposed atoms (distance = 0) are handled and pushed apart."""
        # Two carbon atoms at exactly the same coordinates
        atoms = [
            {
                "atom_number": 1,
                "atom_name": "CA",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "coords": np.array([10.0, 10.0, 10.0]),
                "element": "C",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "coords": np.array([10.0, 10.0, 10.0]),
                "element": "C",
            },
        ]

        tweaked_atoms = PDBValidator._apply_steric_clash_tweak(atoms, min_atom_distance=2.0)

        dist = np.linalg.norm(tweaked_atoms[0]["coords"] - tweaked_atoms[1]["coords"])
        # Should be pushed apart to at least min_atom_distance (2.0)
        # However, due to the way unit_vector is calculated,
        # for EXACT superposition, it might still fail if not handled by a small jitter.
        # Let's see how the current implementation handles it.
        # Actually, if vector is [0,0,0], norm is 0, unit_vector = 0/0 = NaN?
        # Let's check the code: unit_vector = vector / distance.
        # If distance < 1e-6 (or similar), we might need a fallback.

        # If it fails, I'll need to add a jitter to the implementation.
        assert (
            dist > 1.9
        ), f"Exactly superimposed atoms should be pushed apart, but distance is {dist}"

    def test_apply_steric_clash_tweak_multi_chain(self):
        """Test that clashes between different chains are detected and resolved."""
        atoms = [
            {
                "atom_number": 1,
                "atom_name": "CA",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "coords": np.array([0.0, 0.0, 0.0]),
                "element": "C",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "residue_name": "ALA",
                "chain_id": "B",
                "residue_number": 1,
                "coords": np.array([0.5, 0.0, 0.0]),
                "element": "C",
            },
        ]

        tweaked_atoms = PDBValidator._apply_steric_clash_tweak(atoms, min_atom_distance=2.0)
        dist = np.linalg.norm(tweaked_atoms[0]["coords"] - tweaked_atoms[1]["coords"])
        assert dist >= 2.0, f"Multi-chain clash should be resolved to 2.0A, but is {dist}"

    def test_vectorized_vs_reference_random(self):
        """Compare vectorized clash detection against a simple reference for random atoms."""
        num_atoms = 50
        np.random.seed(42)
        coords = np.random.rand(num_atoms, 3) * 10.0
        atoms = []
        for i in range(num_atoms):
            atoms.append(
                {
                    "atom_number": i + 1,
                    "atom_name": "CA",
                    "residue_name": "ALA",
                    "chain_id": "A",
                    "residue_number": i + 1,
                    "coords": coords[i],
                    "element": "C",
                }
            )

        # 1. Vectorized output
        tweaked_vec = PDBValidator._apply_steric_clash_tweak(atoms, min_atom_distance=2.0)

        # 2. Reference Check: Are there any clashes left?
        def count_clashes(parsed_atoms, min_dist=2.0):
            clashes = 0
            for i in range(len(parsed_atoms)):
                for j in range(i + 1, len(parsed_atoms)):
                    d = np.linalg.norm(parsed_atoms[i]["coords"] - parsed_atoms[j]["coords"])
                    if d < min_dist:
                        clashes += 1
            return clashes

        initial_clashes = count_clashes(atoms)
        final_clashes = count_clashes(tweaked_vec)

        print(f"Random test: {initial_clashes} -> {final_clashes} clashes")
        assert final_clashes < initial_clashes, "Vectorized tweak should reduce number of clashes"


if __name__ == "__main__":
    unittest.main()
