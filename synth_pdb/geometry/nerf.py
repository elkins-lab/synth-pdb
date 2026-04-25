"""
NeRF (Natural Extension Reference Frame) geometry implementation.

EDUCATIONAL NOTE - Z-Matrix Construction:
A Z-matrix is a way to represent the geometry of a molecule using internal coordinates:
- Bond Length: Distance between two atoms.
- Bond Angle: Angle formed by three atoms.
- Torsion/Dihedral Angle: Angle between two planes formed by four atoms.

EDUCATIONAL NOTE - NeRF Geometry:
The Natural Extension Reference Frame (NeRF) algorithm provides a high-performance
method for converting internal coordinates back into Cartesian (3D) coordinates.
It ensures mathematical precision and is widely used in protein structure
modeling and molecular dynamics.
"""

import numpy as np

from synth_pdb.geometry._numba import njit


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

    Uses the NeRF method (Natural Extension Reference Frame).
    This implementation is verified to follow IUPAC 0=CIS convention.
    """
    # 1. Convert to float64 for precision
    p1_64 = p1.astype(np.float64)
    p2_64 = p2.astype(np.float64)
    p3_64 = p3.astype(np.float64)

    # 2. Convert to radians
    theta = np.deg2rad(bond_angle_deg)
    chi = np.deg2rad(dihedral_angle_deg)

    # 3. Define local coordinate system
    # Following standard NeRF (e.g. mdtraj, biopython)
    v1 = p1_64 - p2_64  # Vector from P2 to P1
    v2 = p3_64 - p2_64  # Vector from P2 to P3

    # Use small epsilon to avoid division by zero
    norm_v2 = np.linalg.norm(v2)
    u2 = v2 / (norm_v2 + 1e-10)

    # n: Normal to plane P1-P2-P3
    n = np.cross(v1, u2)
    norm_n = np.linalg.norm(n)
    n /= norm_n + 1e-10

    # m: In-plane perpendicular vector (TRANS direction)
    m = np.cross(n, u2)

    # 4. Calculate P4 position in local frame
    p4 = p3_64 + bond_length * (
        -np.cos(theta) * u2 - np.sin(theta) * np.cos(chi) * m - np.sin(theta) * np.sin(chi) * n
    )

    return p4  # type: ignore[no-any-return]


# Alias for backward compatibility
place_atom = position_atom_3d_from_internal_coords
