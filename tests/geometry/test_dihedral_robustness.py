import numpy as np
import pytest

from synth_pdb.geometry.dihedral import calculate_angle, calculate_dihedral


class TestDihedralRobustness:
    """Test suite for dihedral and angle calculation numerical stability."""

    def test_angle_standard(self) -> None:
        """Verify standard angle calculation."""
        c1 = np.array([1.0, 0.0, 0.0])
        c2 = np.array([0.0, 0.0, 0.0])
        c3 = np.array([0.0, 1.0, 0.0])

        # 90 degrees
        angle = calculate_angle(c1, c2, c3)
        assert pytest.approx(angle) == 90.0

    def test_angle_collinear(self) -> None:
        """Verify angle calculation for collinear points."""
        c1 = np.array([1.0, 0.0, 0.0])
        c2 = np.array([0.0, 0.0, 0.0])
        c3 = np.array([-1.0, 0.0, 0.0])

        # 180 degrees
        angle = calculate_angle(c1, c2, c3)
        assert pytest.approx(angle) == 180.0

    def test_angle_zero_length(self) -> None:
        """Verify angle calculation when two points are identical (zero length vector)."""
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([0.0, 0.0, 0.0])
        c3 = np.array([0.0, 1.0, 0.0])

        angle = calculate_angle(c1, c2, c3)
        assert angle == 0.0

    def test_dihedral_standard(self) -> None:
        """Verify standard dihedral calculation."""
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])

        # This is a CIS planar configuration (0 deg)
        dihedral = calculate_dihedral(p1, p2, p3, p4)
        assert pytest.approx(dihedral) == 0.0

        # Trans planar (180 deg)
        p4_trans = np.array([-1.0, 0.0, 0.0])
        dihedral_trans = calculate_dihedral(p1, p2, p3, p4_trans)
        assert abs(dihedral_trans) == 180.0

    def test_dihedral_collinear_singularity(self) -> None:
        """Verify behavior when atoms are collinear."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.0])  # Collinear p1-p2-p3
        p4 = np.array([0.0, 1.0, 0.0])

        # Normal to plane p1-p2-p3 is undefined.
        # Arctan2(0, 0) is 0.0 in numpy.
        dihedral = calculate_dihedral(p1, p2, p3, p4)
        assert not np.isnan(dihedral)

    def test_dihedral_precision(self) -> None:
        """Ensure float64 is used for calculation."""
        p1 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        p2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        p3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p4 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        dihedral = calculate_dihedral(p1, p2, p3, p4)
        assert isinstance(dihedral, float)
