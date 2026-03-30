"""
NeRF (Natural Extension Reference Frame) geometry implementation.
"""

import numpy as np

from synth_pdb.geometry._numba import njit

# EDUCATIONAL NOTE - Z-Matrix Construction
# ----------------------------------------
# Proteins are defined by their "Internal Coordinates" (Z-Matrix):
# 1. Bond Length (distance between two atoms)
# 2. Bond Angle (angle between three atoms)
# 3. Torsion/Dihedral Angle (twist between four atoms)
#
# Our generator builds structures by transitioning from this 1D/2D internal
# representation into 3D Cartesian space.
# This algorithm is the engine of our protein builder, allowing us to
# "walk down" the chain atom-by-atom with mathematical precision.

@njit
def position_atom_3d_from_internal_coords(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    bond_length: float,
    bond_angle_deg: float,
    dihedral_angle_deg: float,
) -> np.ndarray:
    """Calculates the 3D coordinates of a new atom (P4) given the coordinates of three
    preceding atoms (P1, P2, P3) and the internal coordinates.

    # EDUCATIONAL NOTE - NeRF Geometry (Natural Extension Reference Frame)
    # -----------------------------------------------------------------
    # Most protein structures are natively defined by their "Internal Coordinates"
    # (Z-Matrix): Bond Lengths, Bond Angles, and Torsion/Dihedral Angles.
    #
    # To convert these into 3D Cartesian coordinates (X, Y, Z), we use the
    # "NeRF" method (Parsons et al., J. Comput. Chem. 2005).
    #
    # How it works:
    # 1. We define a local coordinate system based on three previous atoms (P1, P2, P3).
    # 2. P3 is the origin (0, 0, 0).
    # 3. The axis b = (P3 - P2) is the primary direction.
    # 4. We use Gram-Schmidt orthogonalization to define the Plane Normal (c) and
    #    the In-Plane Normal (d).
    # 5. The new atom P4 is then "placed" in this local frame using spherical-to-Cartesian
    #    conversion and then transformed back into the global reference frame.
    """
    bond_angle_rad = np.deg2rad(bond_angle_deg)
    dihedral_angle_rad = np.deg2rad(dihedral_angle_deg)

    a = p2 - p1
    b = p3 - p2
    c = np.cross(a, b)
    d = np.cross(c, b)

    # Safe normalization
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    c_norm = np.sqrt(np.sum(c**2))
    d_norm = np.sqrt(np.sum(d**2))

    if a_norm > 0:
        a /= a_norm
    if b_norm > 0:
        b /= b_norm
    if c_norm > 0:
        c /= c_norm
    if d_norm > 0:
        d /= d_norm

    p4 = p3 + bond_length * (
        -b * np.cos(bond_angle_rad)
        + d * np.sin(bond_angle_rad) * np.cos(dihedral_angle_rad)
        + c * np.sin(bond_angle_rad) * np.sin(dihedral_angle_rad)
    )
    return p4  # type: ignore[no-any-return]

# Alias for backward compatibility
place_atom = position_atom_3d_from_internal_coords
