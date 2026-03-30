"""
Dihedral and angle calculation utilities.
"""

import numpy as np

from synth_pdb.geometry._numba import njit

# EDUCATIONAL NOTE - Circular Statistics (The 180/-180 Problem):
# -----------------------------------------------------------
# In protein geometry, torsion angles (Phi, Psi, Omega, Chi) are periodic.
# This introduces a challenge for both math and AI modeling:
#
# 1. The Boundary Artifact: An angle of -179 deg is physically very close to
#    +179 deg, but their arithmetic difference is 358 deg.
# 2. Correct Distance: To find the "real" difference between two angles, we
#    must use: `diff = (a - b + 180) % 360 - 180`.
# 3. AI Loss Functions: Naive Mean Squared Error (MSE) fails on angles because
#    it doesn't understand this wrapping. High-performance models (like
#    AlphaFold) often predict the (Sine, Cosine) of the angle instead,
#    ensuring a smooth, continuous coordinate space.
# 4. Phase Wrapping: In structure generation, "Drift" must be applied carefully
#    to avoid discontinuities at the -180/180 boundary.

@njit
def calculate_angle(coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray) -> float:
    """Calculates the angle (in degrees) formed by three coordinates, with coord2 as the vertex."""
    vec1 = coord1 - coord2
    vec2 = coord3 - coord2

    norm_vec1 = np.sqrt(np.sum(vec1**2))
    norm_vec2 = np.sqrt(np.sum(vec2**2))

    denominator = norm_vec1 * norm_vec2

    if denominator == 0:
        return 0.0

    dot_prod = np.sum(vec1 * vec2)
    cosine_angle = dot_prod / denominator
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))

@njit
def calculate_dihedral(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """Calculates the dihedral angle (in degrees) defined by four points (p1, p2, p3, p4).
    Uses the robust vector-based normal approach (IUPAC convention).
    """
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    # Normals to the two planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Normalize normals
    n1_norm = np.sqrt(np.sum(n1**2))
    n2_norm = np.sqrt(np.sum(n2**2))

    # Safe normalization
    if n1_norm > 0:
        n1 = n1.astype(np.float64) / n1_norm
    else:
        n1 = n1.astype(np.float64) * 0.0

    if n2_norm > 0:
        n2 = n2.astype(np.float64) / n2_norm
    else:
        n2 = n2.astype(np.float64) * 0.0

    # Unit vector along the second bond
    v2_norm = np.sqrt(np.sum(v2**2))
    if v2_norm > 0:
        u2 = v2.astype(np.float64) / v2_norm
    else:
        u2 = v2.astype(np.float64) * 0.0

    # Orthonormal basis in the plane perpendicular to b2
    m1 = np.cross(n1, u2)

    x = np.sum(n1 * n2)
    y = np.sum(m1 * n2)

    return float(-np.degrees(np.arctan2(y, x)))

# Alias for backward compatibility
calculate_dihedral_angle = calculate_dihedral
