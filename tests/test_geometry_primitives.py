import unittest
import numpy as np
from synth_pdb.geometry.nerf import position_atom_3d_from_internal_coords
from synth_pdb.geometry.dihedral import calculate_angle, calculate_dihedral


class TestGeometryPrimitives(unittest.TestCase):
    def test_calculate_angle_basic(self) -> None:
        """Test basic angle calculation."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0)

    def test_calculate_angle_zero_denominator(self) -> None:
        """Test angle calculation with zero denominator (coincident points)."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(p1, p2, p3)
        self.assertEqual(angle, 0.0)

    def test_calculate_dihedral_basic(self) -> None:
        """Test basic dihedral calculation (trans)."""
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])
        dihedral = calculate_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(dihedral, 0.0)

    def test_calculate_dihedral_cis(self) -> None:
        """Test basic dihedral calculation (cis)."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        dihedral = calculate_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(dihedral, 0.0)

        # Proper CIS test
        p1 = np.array([1, 1, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([1, 0, 0])
        p4 = np.array([2, 1, 0])
        # This is essentially a zig-zag in XY plane.
        # Let's try simpler:
        p1 = np.array([1, 1, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([1, 0, 0])
        p4 = np.array([0, 1, 0])
        # p1-p2 vector: [1, 1, 0]
        # p2-p3 vector: [1, 0, 0]
        # p3-p4 vector: [-1, 1, 0]
        # Both p1 and p4 are on same side of p2-p3 line?
        # Actually let's just use 90 degrees.
        p1 = np.array([0, 1, 1])
        p2 = np.array([0, 0, 1])
        p3 = np.array([0, 0, 0])
        p4 = np.array([1, 0, 0])
        # plane 1 (p1,p2,p3): X=0
        # plane 2 (p2,p3,p4): Y=0
        dihedral = calculate_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(abs(dihedral), 90.0)

    def test_nerf_placement(self) -> None:
        """Test NeRF atom placement and verify consistency with dihedral calculation."""
        p1 = np.array([-1.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])

        bond_length = 1.5
        bond_angle = 110.0
        dihedral = 60.0

        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, bond_length, bond_angle, dihedral)

        # Verify bond length
        calc_bond = np.linalg.norm(p4 - p3)
        self.assertAlmostEqual(calc_bond, bond_length)

        # Verify bond angle
        calc_angle = calculate_angle(p2, p3, p4)
        self.assertAlmostEqual(calc_angle, bond_angle)

        # Verify dihedral
        calc_dihedral = calculate_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(calc_dihedral, dihedral)

    def test_nerf_collinear_p2_p3(self) -> None:
        """Test NeRF with near-zero vector for P2-P3 to check robustness."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1e-12, 0.0, 0.0])

        # Should not crash due to 1e-10 epsilon in nerf.py
        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, 60.0)
        self.assertEqual(p4.shape, (3,))


if __name__ == "__main__":
    unittest.main()
