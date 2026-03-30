"""
Vectorized geometry operations for high-performance batch processing.
"""

from typing import Tuple, cast

import numpy as np

# EDUCATIONAL NOTE - SIMD & Parallel Geometry:
# -------------------------------------------
# Traditional biology code uses "Serial Geometry" ($O(B \times L)$).
# To place atoms for $B$ structures of length $L$, it loops $B$ times.
#
# BatchedGenerator uses Single Instruction, Multiple Data (SIMD) logic:
# 1. Broad Geometry: We treat the coordinates as a massive block of numbers
#    rather than individual XYZ points.
# 2. Vector Units: Hardware like the M4's AMX or a GPU's CUDA cores can execute
#    one operation (e.g., a cross product) across thousands of data points at once.
# 3. Efficiency: By avoiding the Python interpreter loop for each structure, we
#    reach throughput levels required for "Foundation Model" training in proteomics.

def superimpose_batch(sources: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Kabsch algorithm to find the optimal rotation and translation
    that aligns a batch of source point sets to target point sets.

    EDUCATIONAL NOTE - Vectorized Kabsch Algorithm:
    ----------------------------------------------
    The Kabsch algorithm finds the optimal rotation matrix that minimizes the
    Root Mean Square Deviation (RMSD) between two sets of points.

    By vectorizing this across B structures:
    1. Centering: We calculate B centroids simultaneously and subtract them.
    2. Covariance: We compute B covariance matrices (3x3) using batch matrix multiplication.
    3. SVD: We perform B Singular Value Decompositions in a single call.
    4. Rotation Correction: We handle the batch-wise determinant to avoid
       reflections (ensuring a right-handed coordinate system).
    """
    b = sources.shape[0]

    # 1. Centroids
    source_centroid = np.mean(sources, axis=1, keepdims=True)  # (B, 1, 3)
    target_centroid = np.mean(targets, axis=1, keepdims=True)  # (B, 1, 3)

    s_centered = sources - source_centroid
    t_centered = targets - target_centroid

    # 2. Covariance Matrix (B, 3, 3)
    cov = np.matmul(s_centered.transpose(0, 2, 1), t_centered)

    # 3. SVD (Batched)
    u, s, vt = np.linalg.svd(cov)

    # 4. Optimal Rotation Matrix: V * U^T
    v = np.transpose(vt, (0, 2, 1))
    u_t = np.transpose(u, (0, 2, 1))
    rotations = np.matmul(v, u_t)

    # 5. Handle Reflections (Negative Determinant)
    det = np.linalg.det(rotations)

    correction = np.repeat(np.eye(3)[np.newaxis, :, :], b, axis=0)
    correction[:, 2, 2] = np.sign(det)

    rotations = np.matmul(v, np.matmul(correction, u_t))

    # Translation = Target Centroid - (Rot * Source Centroid)
    translations = target_centroid.squeeze(1) - np.matmul(
        rotations, source_centroid.transpose(0, 2, 1)
    ).squeeze(2)

    return translations, rotations


def position_atoms_batch(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    bond_lengths: np.ndarray,
    bond_angles_deg: np.ndarray,
    dihedral_angles_deg: np.ndarray,
) -> np.ndarray:
    """Vectorized version of the NeRF algorithm for large batches of structures.

    EDUCATIONAL NOTE - GPU-First Operations:
    ---------------------------------------
    On modern hardware (Apple M4 AMX, NVIDIA Tensor Cores), serial loops are
    extremely inefficient. By vectorizing the math into large matrix operations:
    1. Memory bandwidth is maximized via contiguous array access.
    2. SIMD units perform the same calculation across multiple samples simultaneously.
    3. Hardware acceleration (Accelerate/MPS/Metal) can be leveraged automatically
       by numpy and high-level frameworks.
    """
    # Convert angles to radians
    angles_rad = np.deg2rad(bond_angles_deg)
    dihedrals_rad = np.deg2rad(dihedral_angles_deg)

    # Calculate relative vectors
    a = p2 - p1
    b = p3 - p2

    # Batch cross products
    c = np.cross(a, b, axis=-1)
    d = np.cross(c, b, axis=-1)

    # Normalize vectors (Batch-wise)
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return cast(np.ndarray, v / norm)

    b = normalize(b)
    c = normalize(c)
    d = normalize(d)

    # Reshape lengths for broadcasting (b, 1)
    length = bond_lengths.reshape(-1, 1)

    # NeRF Coordinate Transformation
    p4 = p3 + length * (
        -b * np.cos(angles_rad).reshape(-1, 1)
        + d * (np.sin(angles_rad) * np.cos(dihedrals_rad)).reshape(-1, 1)
        + c * (np.sin(angles_rad) * np.sin(dihedrals_rad)).reshape(-1, 1)
    )

    return cast(np.ndarray, p4)


def batched_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Vectorized calculation of angles for batches of point triplets."""
    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)

    cos = dot / (norm1 * norm2 + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)

    return np.degrees(np.arccos(cos))  # type: ignore[no-any-return]


def batched_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Vectorized calculation of dihedral angles for batches of point quadruplets."""
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-9

    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v, axis=-1) * w, axis=-1)

    return np.degrees(np.arctan2(y, x))  # type: ignore[no-any-return]
