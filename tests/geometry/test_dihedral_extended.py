"""
Extended property-based tests for dihedral and angle calculations.
"""

import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from synth_pdb.geometry.dihedral import calculate_angle, calculate_dihedral


# Strategy for coordinates: 3D points with reasonable bounds
coords_st = arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(min_value=-1000.0, max_value=1000.0),
)


@settings(deadline=None)
@given(p1=coords_st, p2=coords_st, p3=coords_st)
def test_angle_properties(p1, p2, p3):
    """
    Properties:
    1. Angle is symmetric: angle(p1, p2, p3) == angle(p3, p2, p1)
    2. Angle is always between 0 and 180 degrees.
    """
    angle1 = calculate_angle(p1, p2, p3)
    angle2 = calculate_angle(p3, p2, p1)

    assert np.isclose(angle1, angle2, atol=1e-7)
    assert 0.0 <= angle1 <= 180.0000001


@settings(deadline=None)
@given(p1=coords_st, p2=coords_st, p3=coords_st, p4=coords_st)
def test_dihedral_properties(p1, p2, p3, p4):
    """
    Properties:
    1. Dihedral is always between -180 and 180 degrees.
    2. Inverting the sequence yields the IDENTICAL dihedral: dih(1,2,3,4) == dih(4,3,2,1)
    """
    # Filter degenerate cases that make dihedral undefined
    if (
        np.linalg.norm(p2 - p1) < 1e-4
        or np.linalg.norm(p3 - p2) < 1e-4
        or np.linalg.norm(p4 - p3) < 1e-4
    ):
        return

    # Check for collinearity
    v1 = p1 - p2
    v2 = p3 - p2
    v3 = p4 - p3
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-4:
        return

    # Normal to planes should be non-zero
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    if np.linalg.norm(n1) < 1e-4 or np.linalg.norm(n2) < 1e-4:
        return

    dih1 = calculate_dihedral(p1, p2, p3, p4)
    dih2 = calculate_dihedral(p4, p3, p2, p1)

    assert -180.0000001 <= dih1 <= 180.0000001

    # Inversion property: dih(1,2,3,4) == dih(4,3,2,1)
    # Use circular distance for the check
    diff = (dih1 - dih2 + 180) % 360 - 180
    assert abs(diff) < 1e-4


@settings(deadline=None)
@given(p1=coords_st, p2=coords_st, p3=coords_st, p4=coords_st, shift=coords_st)
def test_dihedral_translation_invariance(p1, p2, p3, p4, shift):
    """Dihedral should be invariant to translation."""
    # Filter degenerate cases
    if np.linalg.norm(p3 - p2) < 1e-4:
        return
    v1 = p1 - p2
    v2 = p3 - p2
    v3 = p4 - p3
    if np.linalg.norm(np.cross(v1, v2)) < 1e-4 or np.linalg.norm(np.cross(v2, v3)) < 1e-4:
        return

    dih_orig = calculate_dihedral(p1, p2, p3, p4)
    dih_shifted = calculate_dihedral(p1 + shift, p2 + shift, p3 + shift, p4 + shift)

    # Use circular difference
    diff = (dih_orig - dih_shifted + 180) % 360 - 180
    assert abs(diff) < 1e-4


def test_angle_zero_length():
    """Verify that zero length vectors don't cause NaNs."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])

    angle = calculate_angle(p1, p2, p3)
    assert angle == 0.0
    assert not np.isnan(angle)


def test_dihedral_collinear():
    """Verify that collinear points don't cause NaNs in dihedral."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])
    p4 = np.array([3.0, 0.0, 0.0])

    dih = calculate_dihedral(p1, p2, p3, p4)
    assert not np.isnan(dih)
