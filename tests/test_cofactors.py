import unittest

import biotite.structure as struc
import numpy as np

from synth_pdb.cofactors import add_metal_ion, find_metal_binding_sites


class TestCofactors(unittest.TestCase):
    def setUp(self):
        # Create a more robust structure with a C2H2 zinc finger motif.
        # We will have CYS, HIS, CYS, HIS residues.
        # Ligands will be CYS-SG, HIS-NE2, CYS-SG, HIS-NE2.
        # We'll also include HIS-ND1 atoms but place them far away.
        self.structure = struc.AtomArray(32)

        # Residue 1: CYS
        self.structure.res_name[:4] = "CYS"
        self.structure.atom_name[:4] = ["N", "CA", "C", "SG"]
        self.structure.res_id[:4] = 1

        # Residue 2: HIS
        self.structure.res_name[4:14] = "HIS"
        self.structure.atom_name[4:14] = [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
        ]
        self.structure.res_id[4:14] = 2

        # Residue 3: CYS
        self.structure.res_name[14:18] = "CYS"
        self.structure.atom_name[14:18] = ["N", "CA", "C", "SG"]
        self.structure.res_id[14:18] = 3

        # Residue 4: HIS
        self.structure.res_name[18:28] = "HIS"
        self.structure.atom_name[18:28] = [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
        ]
        self.structure.res_id[18:28] = 4

        # Add a dummy residue to make it 32 atoms
        self.structure.res_name[28:] = "DUM"
        self.structure.atom_name[28:] = ["D1", "D2", "D3", "D4"]
        self.structure.res_id[28:] = 5

        # Initialize all coordinates to be far apart
        self.structure.coord = np.arange(32 * 3).reshape(32, 3) * 100.0

        # Place the desired ligands in a nice tight cluster
        self.structure.coord[3] = [0, 0, 0]  # CYS 1 SG (index 3)
        self.structure.coord[13] = [2, 0, 0]  # HIS 2 NE2 (index 13)
        self.structure.coord[17] = [1, 1.732, 0]  # CYS 3 SG (index 17)
        self.structure.coord[27] = [1, 0.577, 1.633]  # HIS 4 NE2 (index 27)

        # Place other potential ligands far away
        self.structure.coord[10] = [50, 50, 50]  # HIS 2 ND1 (index 10)
        self.structure.coord[24] = [-50, -50, -50]  # HIS 4 ND1 (index 24)

        self.structure.chain_id[:] = "A"
        self.structure.hetero[:] = False
        # A simplified element list
        self.structure.element = np.array(
            ["N", "C", "C", "S"]
            + ["N", "C", "C", "O", "C", "C", "N", "C", "C", "N"]
            + ["N", "C", "C", "S"]
            + ["N", "C", "C", "O", "C", "C", "N", "C", "C", "N"]
            + ["X", "X", "X", "X"]
        )

    def test_find_metal_binding_sites_success(self):
        """Test finding a C2H2 zinc finger."""
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["type"], "ZN")
        self.assertEqual(len(sites[0]["ligand_indices"]), 4)
        # Check if the correct ligand atoms are identified
        found_indices = sorted(sites[0]["ligand_indices"])
        expected_indices = sorted([3, 13, 17, 27])  # SG, NE2, SG, NE2
        self.assertListEqual(found_indices, expected_indices)

    def test_find_metal_binding_sites_no_site(self):
        """Test with a structure that has no valid binding site."""
        # Scatter the ligands far apart
        self.structure.coord[13] = [20, 0, 0]
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 0)

    def test_add_metal_ion(self):
        """Test adding a zinc ion to a found site."""
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 1)

        new_structure = add_metal_ion(self.structure, sites[0])

        # Check that one atom was added
        self.assertEqual(len(new_structure), len(self.structure) + 1)

        # Check the new atom is a Zinc ion
        ion = new_structure[-1]
        self.assertEqual(ion.res_name, "ZN")
        self.assertEqual(ion.atom_name, "ZN")
        self.assertEqual(ion.element, "ZN")
        self.assertTrue(ion.hetero)

        # Check the ion is at the centroid of the ligands
        ligand_coords = self.structure.coord[[3, 13, 17, 27]]
        expected_centroid = np.mean(ligand_coords, axis=0)
        np.testing.assert_allclose(ion.coord, expected_centroid, atol=1e-6)

    def test_multi_chain_coordination(self):
        """Verify that coordination can occur between atoms on different chains."""
        # Setup: 2 ligands on Chain A, 2 on Chain B
        struct = struc.AtomArray(4)
        struct.res_name = np.array(["CYS", "CYS", "HIS", "HIS"])
        struct.atom_name = np.array(["SG", "SG", "NE2", "NE2"])
        struct.res_id = np.array([1, 2, 1, 2])
        struct.chain_id = np.array(["A", "A", "B", "B"])
        struct.coord = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
        struct.hetero = np.array([False] * 4)

        sites = find_metal_binding_sites(struct, distance_threshold=5.0)
        self.assertEqual(len(sites), 1, "Should coordinate across chains")
        self.assertEqual(len(sites[0]["ligand_indices"]), 4)

    def test_insufficient_unique_residues(self):
        """Ensure 4 atoms from only 3 unique residues do NOT form a site."""
        struct = struc.AtomArray(4)
        # Residue 1 has two coordination atoms (ND1 and NE2)
        struct.res_name = np.array(["HIS", "HIS", "CYS", "CYS"])
        struct.atom_name = np.array(["ND1", "NE2", "SG", "SG"])
        struct.res_id = np.array([1, 1, 2, 3])
        struct.chain_id = np.array(["A", "A", "A", "A"])
        struct.coord = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [0, 1, 0]])
        struct.hetero = np.array([False] * 4)

        sites = find_metal_binding_sites(struct, distance_threshold=5.0)
        self.assertEqual(len(sites), 0, "Should not coordinate with only 3 unique residues")

    def test_overlapping_sites_tightest_first(self):
        """Verify that the algorithm picks the tightest cluster when sites overlap."""
        # 5 ligands total. Atom 0-3 form a tight cluster. Atom 0-2 + 4 form a loose cluster.
        struct = struc.AtomArray(5)
        struct.res_name = np.array(["CYS"] * 5)
        struct.atom_name = np.array(["SG"] * 5)
        struct.res_id = np.array([1, 2, 3, 4, 5])
        struct.chain_id = np.array(["A"] * 5)
        struct.coord = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],  # Tight partner
                [10, 10, 10],  # Loose partner
            ]
        )
        struct.hetero = np.array([False] * 5)

        sites = find_metal_binding_sites(struct, distance_threshold=20.0)
        self.assertEqual(len(sites), 1)
        # Should have picked indices 0, 1, 2, 3
        self.assertIn(3, sites[0]["ligand_indices"])
        self.assertNotIn(4, sites[0]["ligand_indices"])

    def test_multi_atom_residue_logic(self):
        """Test a hypothetical residue with 3 potential ligands (triple-counting check)."""
        # Residue 1 has 3 atoms that match the ligand mask
        struct = struc.AtomArray(6)
        struct.res_name = np.array(["HIS", "HIS", "HIS", "CYS", "CYS", "CYS"])
        struct.atom_name = np.array(
            ["ND1", "NE2", "CG", "SG", "SG", "SG"]
        )  # CG is not in mask usually, let's use valid ones
        # Actually HIS only has ND1 and NE2. Let's force them via mask in a custom way if needed,
        # or just use 2 HIS and 2 CYS but one HIS has both.

        # Reset to real scenario: 2 atoms in HIS 1, 1 atom in CYS 2, 1 atom in CYS 3
        # Total 4 atoms, but only 3 unique residues.
        struct.res_name = np.array(["HIS", "HIS", "CYS", "CYS", "ALA", "ALA"])
        struct.atom_name = np.array(["ND1", "NE2", "SG", "SG", "CA", "CA"])
        struct.res_id = np.array([1, 1, 2, 3, 4, 5])
        struct.chain_id = np.array(["A", "A", "A", "A", "A", "A"])
        struct.coord = np.array(
            [
                [0, 0, 0],  # HIS 1 ND1
                [0.1, 0, 0],  # HIS 1 NE2 (very close)
                [1, 0, 0],  # CYS 2
                [0, 1, 0],  # CYS 3
                [10, 10, 10],
                [11, 11, 11],
            ]
        )
        struct.hetero = np.array([False] * 6)

        sites = find_metal_binding_sites(struct, distance_threshold=20.0)
        self.assertEqual(
            len(sites),
            0,
            "Should NOT form a site with only 3 unique residues even if 4 candidate atoms exist",
        )

        # Now add a 4th unique residue
        struct2 = struc.AtomArray(7)
        struct2.res_name = np.array(["HIS", "HIS", "CYS", "CYS", "CYS", "ALA", "ALA"])
        struct2.atom_name = np.array(["ND1", "NE2", "SG", "SG", "SG", "CA", "CA"])
        struct2.res_id = np.array([1, 1, 2, 3, 4, 5, 6])
        struct2.chain_id = np.array(["A", "A", "A", "A", "A", "A", "A"])
        struct2.coord = np.array(
            [
                [0, 0, 0],  # HIS 1 ND1
                [0.1, 0, 0],  # HIS 1 NE2
                [1, 0, 0],  # CYS 2
                [0, 1, 0],  # CYS 3
                [1, 1, 0],  # CYS 4
                [10, 10, 10],
                [11, 11, 11],
            ]
        )
        struct2.hetero = np.array([False] * 7)

        sites2 = find_metal_binding_sites(struct2, distance_threshold=20.0)
        self.assertEqual(
            len(sites2), 1, "Should form a site now that 4 unique residues are present"
        )
        self.assertEqual(len(sites2[0]["ligand_indices"]), 4)

        # Verify it picked the BEST atom from HIS 1 (the one closest to the centroid or seed)
        # If we use HIS 1 ND1 (index 0) as seed, distance to NE2 (index 1) is 0.1, but it should
        # only pick ONE of them.
        ligands = sites2[0]["ligand_indices"]
        self.assertTrue((0 in ligands) ^ (1 in ligands), "Should pick exactly one atom from HIS 1")


if __name__ == "__main__":
    unittest.main()
