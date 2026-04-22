"""
Optimal superposition of structures using the Kabsch algorithm.

NOTE ON ATOM SELECTION:
These functions operate on raw Nx3 coordinate arrays and are atom-agnostic.
For standard protein 'Backbone Superposition', coordinates should be
pre-filtered to include only backbone heavy atoms (typically N, CA, C).
For overall fold alignment, C-alpha (CA) only is often used.
"""

import logging
from typing import List

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def kabsch_superposition(
    P: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]  # noqa: N803
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate optimal rotation and translation to superimpose P onto Q.

    Uses the Kabsch algorithm (1976, 1978) with SVD decomposition to find
    the optimal rotation matrix and translation vector that minimizes
    the RMSD between two sets of coordinates.

    Note: This function assumes the input arrays P and Q are already
    filtered for the desired atoms (e.g., N, CA, C) and are in 1-to-1
    correspondence.

    Args:
        P: Nx3 array of coordinates (mobile structure)
        Q: Nx3 array of coordinates (reference structure)

    Returns:
        Tuple of (rotation_matrix, translation_vector)
        - rotation_matrix: 3x3 rotation matrix
        - translation_vector: 3-element translation vector

        To apply transformation: P_aligned = (R @ P.T).T + t

    Raises:
        ValueError: If P and Q have different shapes or not Nx3
    """
    # Validate input
    if P.shape != Q.shape:
        logger.error(f"Shape mismatch: P.shape={P.shape}, Q.shape={Q.shape}")
        raise ValueError(f"Coordinate arrays must have same shape. Got P: {P.shape}, Q: {Q.shape}")

    if len(P.shape) != 2 or P.shape[1] != 3:
        logger.error(f"Invalid shape: {P.shape}")
        raise ValueError(f"Coordinates must be Nx3 arrays. Got shape: {P.shape}")

    # Step 1: Center both structures
    if P.size == 0 or Q.size == 0:
        logger.warning("kabsch_superposition: Empty input arrays.")
        return np.array([]), np.array([])
    P_center = P.mean(axis=0)  # noqa: N806
    Q_center = Q.mean(axis=0)  # noqa: N806

    P_centered = P - P_center  # noqa: N806
    Q_centered = Q - Q_center  # noqa: N806

    # Step 2: Compute covariance matrix
    H = P_centered.T @ Q_centered  # noqa: N806

    # Check for numerical issues in H
    if not np.all(np.isfinite(H)):
        logger.warning("kabsch_superposition: H contains non-finite values.")
        return np.eye(3), np.zeros(3)

    # Step 3: SVD decomposition
    try:
        U, S, Vt = np.linalg.svd(H)  # noqa: N806
    except np.linalg.LinAlgError as e:
        logger.error(f"kabsch_superposition: SVD failed to converge: {e}")
        return np.eye(3), np.zeros(3)

    # Step 4: Compute rotation matrix
    try:
        d = np.linalg.det(Vt.T @ U.T)
    except np.linalg.LinAlgError:
        d = 1.0

    if abs(d) < 1e-12:
        d_corr = 1.0 if d >= 0 else -1.0
    else:
        d_corr = 1.0 if d > 0 else -1.0

    diag = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d_corr]])

    R = Vt.T @ diag @ U.T  # noqa: N806

    # Final check on R
    if not np.all(np.isfinite(R)):
        logger.warning("kabsch_superposition: Input matrices contain non-finite values.")
        return np.eye(3), np.zeros(3)

    # Step 5: Compute translation
    t = Q_center - R @ P_center  # noqa: N806

    return R, t


def apply_transformation(
    coords: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Apply rotation and translation to coordinates.

    Args:
        coords: Nx3 array of coordinates
        rotation: 3x3 rotation matrix
        translation: 3-element translation vector

    Returns:
        Nx3 array of transformed coordinates

    Examples:
        >>> import numpy as np
        >>> coords = np.array([[1., 0., 0.], [0., 1., 0.]])
        >>> R = np.eye(3)  # Identity rotation
        >>> t = np.array([1., 2., 3.])
        >>> transformed = apply_transformation(coords, R, t)
        >>> np.allclose(transformed, coords + t)
        True
    """
    return (rotation @ coords.T).T + translation


def superimpose_structures(
    mobile: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Superimpose mobile structure onto reference structure.

    Convenience function that combines kabsch_superposition and apply_transformation.

    Args:
        mobile: Nx3 array of coordinates to be moved
        reference: Nx3 array of reference coordinates

    Returns:
        Nx3 array of superimposed mobile coordinates

    Examples:
        >>> import numpy as np
        >>> mobile = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> reference = np.array([[5., 3., -2.], [6., 3., -2.]])
        >>> aligned = superimpose_structures(mobile, reference)
    """
    R, t = kabsch_superposition(mobile, reference)  # noqa: N806
    return apply_transformation(mobile, R, t)


def find_medoid(coords_list: List[npt.NDArray[np.float64]], superimpose: bool = True) -> int:
    """
    Find medoid structure (most representative) from ensemble.

    The medoid is the structure that minimizes the sum of RMSD values
    to all other structures in the ensemble. This is the recommended
    way to select a representative structure from an NMR ensemble.

    Args:
        coords_list: List of Nx3 coordinate arrays
        superimpose: If True, perform optimal superposition before RMSD calculation

    Returns:
        Index of the medoid structure

    Examples:
        >>> import numpy as np
        >>> c1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> c2 = np.array([[0., 0., 0.], [1., 0., 0.]])  # Identical to c1
        >>> c3 = np.array([[10., 10., 10.], [11., 10., 10.]])  # Far away
        >>> medoid_idx = find_medoid([c1, c2, c3])
        >>> medoid_idx in [0, 1]  # Should be c1 or c2
        True
    """
    from synth_pdb.geometry.rmsd import calculate_pairwise_rmsd

    rmsd_matrix = calculate_pairwise_rmsd(coords_list, superimpose=superimpose)
    sum_rmsds = rmsd_matrix.sum(axis=1)
    return int(np.argmin(sum_rmsds))
