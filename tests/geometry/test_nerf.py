"""
Tests for NeRF geometry implementation.
"""

import numpy as np

from synth_pdb.geometry.nerf import position_atom_3d_from_internal_coords


def test_position_atom_3d_consistency():
    """Test position_atom_3d_from_internal_coords returns consistent values."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p3 = np.array([2.0, 1.0, 0.0])

    # Internal coords for a regular geometry
    b_len = 1.5
    b_ang = 110.0
    dihedral = 180.0

    pos = position_atom_3d_from_internal_coords(p1, p2, p3, b_len, b_ang, dihedral)
    assert pos.shape == (3,)
    assert not np.any(np.isnan(pos))
