"""
Root Mean Square Deviation (RMSD) and ensemble average calculations.

NOTE ON ATOM SELECTION:
These functions operate on raw Nx3 coordinate arrays and are atom-agnostic.
For standard protein 'Backbone RMSD', coordinates should be pre-filtered to
include only backbone heavy atoms (typically N, CA, C). For overall fold
comparison, C-alpha (CA) only is often used.
"""

import logging
from typing import cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def calculate_rmsd(P: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]) -> float:  # noqa: N803
    """
    Calculate Root Mean Square Deviation between two sets of coordinates.

    The RMSD is calculated as:
        RMSD = sqrt(mean(sum((P - Q)^2, axis=1)))

    Note: This function assumes the input arrays P and Q are already
    aligned/superimposed and filtered for the desired atoms (e.g., N, CA, C).

    Args:
        P: Nx3 array of coordinates (first structure)
        Q: Nx3 array of coordinates (second structure)

    Returns:
        RMSD value in Angstroms

    Examples:
        >>> import numpy as np
        >>> coords1 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> coords2 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> calculate_rmsd(coords1, coords2)
        0.0
    """
    if P.shape != Q.shape:
        logger.error(f"Shape mismatch: P.shape={P.shape}, Q.shape={Q.shape}")
        raise ValueError(f"Coordinate arrays must have same shape. Got P: {P.shape}, Q: {Q.shape}")

    if len(P.shape) != 2 or P.shape[1] != 3:
        logger.error(f"Invalid shape: {P.shape}")
        raise ValueError(f"Coordinates must be Nx3 arrays. Got shape: {P.shape}")

    if P.shape[0] == 0:
        logger.warning("calculate_rmsd: Zero points provided.")
        return 0.0

    # Calculate squared differences
    diff = P - Q
    squared_diff = np.sum(diff**2, axis=1)
    rmsd = float(np.sqrt(np.mean(squared_diff)))
    return rmsd


def calculate_pairwise_rmsd(
    coords_list: list[npt.NDArray[np.float64]], superimpose: bool = False
) -> npt.NDArray[np.float64]:
    """
    Calculate pairwise RMSD matrix for multiple structures.

    Args:
        coords_list: List of Nx3 coordinate arrays
        superimpose: If True, perform optimal superposition before RMSD calculation

    Returns:
        MxM symmetric matrix of pairwise RMSD values,
        where M is the number of structures

    Examples:
        >>> import numpy as np
        >>> c1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> c2 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> c3 = np.array([[1., 0., 0.], [2., 0., 0.]])
        >>> rmsd_matrix = calculate_pairwise_rmsd([c1, c2, c3])
        >>> rmsd_matrix.shape
        (3, 3)
        >>> float(rmsd_matrix[0, 1])  # c1 vs c2
        0.0
    """
    from synth_pdb.geometry.superposition import kabsch_superposition

    n_structures = len(coords_list)
    rmsd_matrix = np.zeros((n_structures, n_structures))

    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            if superimpose:
                R, t = kabsch_superposition(coords_list[i], coords_list[j])  # noqa: N806
                aligned_i = (R @ coords_list[i].T).T + t
                rmsd = calculate_rmsd(aligned_i, coords_list[j])
            else:
                rmsd = calculate_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd  # Symmetric
    return rmsd_matrix


def calculate_average_coords(
    coords_list: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Calculate average (centroid) coordinates from ensemble.

    Args:
        coords_list: List of Nx3 coordinate arrays from ensemble

    Returns:
        Nx3 array of average coordinates

    Examples:
        >>> import numpy as np
        >>> c1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> c2 = np.array([[2., 0., 0.], [3., 0., 0.]])
        >>> avg = calculate_average_coords([c1, c2])
        >>> avg
        array([[1., 0., 0.],
               [2., 0., 0.]])
    """
    if not coords_list:
        return np.array([]).reshape(0, 3)

    arr = np.array(coords_list)
    if arr.size == 0:
        logger.warning("calculate_average_coords: Input array is empty.")
        return np.array([]).reshape(0, 3)

    avg = cast(npt.NDArray[np.float64], np.mean(arr, axis=0))
    return avg


def calculate_rmsd_to_average(
    coords_list: list[npt.NDArray[np.float64]],
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Calculate RMSD of each structure to the average structure.

    This is commonly reported for NMR ensembles.

    Args:
        coords_list: List of Nx3 coordinate arrays

    Returns:
        Tuple of (average_rmsd, average_coords)
        - average_rmsd: Mean RMSD across all structures
        - average_coords: Nx3 array of average coordinates

    Examples:
        >>> import numpy as np
        >>> c1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> c2 = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> avg_rmsd, avg_coords = calculate_rmsd_to_average([c1, c2])
        >>> float(avg_rmsd)
        0.0
    """
    avg_coords = calculate_average_coords(coords_list)
    if avg_coords.size == 0 or not coords_list:
        logger.warning("calculate_rmsd_to_average: No coordinates provided.")
        return float("nan"), avg_coords
    rmsds = [calculate_rmsd(coords, avg_coords) for coords in coords_list]
    if len(rmsds) == 0:
        logger.warning("calculate_rmsd_to_average: No RMSDs calculated.")
        return float("nan"), avg_coords
    avg_rmsd = float(np.mean(rmsds))
    return avg_rmsd, avg_coords
