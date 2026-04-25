import numpy as np

from synth_pdb.geometry.nerf import position_atom_3d_from_internal_coords


class TestNerfRobustness:
    """Test suite for NeRF coordinate generation numerical stability."""

    def test_standard_case(self) -> None:
        """Verify standard placement logic."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        # Place P4 at 90 deg bond angle, 0 deg dihedral (CIS), 1.0 length
        # Should be at [1, 1, 0]
        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 0.0)
        np.testing.assert_allclose(p4, np.array([1.0, 1.0, 0.0]), atol=1e-7)

    def test_collinear_atoms_singularity(self) -> None:
        """
        Verify behavior when atoms are collinear (bond angle = 180).
        With epsilon added, this should no longer produce NaNs.
        """
        p1 = np.array([-1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])

        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 0.0)
        assert not np.any(np.isnan(p4))
        assert not np.any(np.isinf(p4))

    def test_identical_atoms_singularity(self) -> None:
        """Verify behavior when p2 and p3 are identical."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])

        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 0.0)
        assert not np.any(np.isnan(p4))
        assert not np.any(np.isinf(p4))

    def test_extreme_angles(self) -> None:
        """Verify stability at extreme dihedral and bond angles."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        # Test 0, 360, 720 degrees
        p4_0 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 0.0)
        p4_360 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 360.0)
        np.testing.assert_allclose(p4_0, p4_360, atol=1e-7)

        # Test very small bond angle
        p4_small = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 0.0001, 0.0)
        assert not np.any(np.isnan(p4_small))

    def test_precision_consistency(self) -> None:
        """Ensure float64 is used internally for precision."""
        p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 90.0, 0.0)
        assert p4.dtype == np.float64
