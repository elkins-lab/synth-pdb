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
    vec1 = coord1.astype(np.float64) - coord2.astype(np.float64)
    vec2 = coord3.astype(np.float64) - coord2.astype(np.float64)

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
    Uses the robust Praxeolitic formula for numerical stability and IUPAC convention.
    """
    b0 = -1.0 * (p2.astype(np.float64) - p1.astype(np.float64))
    b1 = p3.astype(np.float64) - p2.astype(np.float64)
    b2 = p4.astype(np.float64) - p3.astype(np.float64)

    # Normalize b1
    b1_norm = np.sqrt(np.sum(b1**2))
    if b1_norm > 0:
        b1 = b1 / b1_norm

    # v = orthogonal component of b0 with respect to b1
    # v = b0 - proj_b1(b0)
    v = b0 - np.sum(b0 * b1) * b1
    # w = orthogonal component of b2 with respect to b1
    # w = b2 - proj_b1(b2)
    w = b2 - np.sum(b2 * b1) * b1

    # x = dot product of v and w
    x = np.sum(v * w)
    # y = dot product of cross(b1, v) and w
    y = np.sum(np.cross(b1, v) * w)

    return float(np.degrees(np.arctan2(y, x)))

# Alias for backward compatibility
calculate_dihedral_angle = calculate_dihedral
