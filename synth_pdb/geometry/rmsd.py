"""
Root Mean Square Deviation (RMSD) and ensemble average calculations.
"""

import logging
from typing import List, cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def calculate_rmsd(P: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]) -> float:
    """
    Calculate Root Mean Square Deviation between two sets of coordinates.
    """
    if P.shape != Q.shape:
        logger.error(f"Shape mismatch: P.shape={P.shape}, Q.shape={Q.shape}")
        raise ValueError(
            f"Coordinate arrays must have same shape. Got P: {P.shape}, Q: {Q.shape}"
        )

    if len(P.shape) != 2 or P.shape[1] != 3:
        logger.error(f"Invalid shape: {P.shape}")
        raise ValueError(f"Coordinates must be Nx3 arrays. Got shape: {P.shape}")

    if len(P) == 0:
        return 0.0

    if P.size == 0 or Q.size == 0:
        logger.warning("calculate_rmsd: Empty input arrays.")
        return float("nan")

    # Calculate squared differences
    diff = P - Q
    squared_diff = np.sum(diff**2, axis=1)
    if squared_diff.size == 0:
        logger.warning("calculate_rmsd: squared_diff is empty. Returning np.nan.")
        return float("nan")
    rmsd = float(np.sqrt(np.mean(squared_diff)))
    return rmsd


def calculate_pairwise_rmsd(
    coords_list: List[npt.NDArray[np.float64]], superimpose: bool = False
) -> npt.NDArray[np.float64]:
    """
    Calculate pairwise RMSD matrix for multiple structures.
    """
    from synth_pdb.geometry.superposition import kabsch_superposition

    n_structures = len(coords_list)
    rmsd_matrix = np.zeros((n_structures, n_structures))

    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            if superimpose:
                R, t = kabsch_superposition(coords_list[i], coords_list[j])
                aligned_i = (R @ coords_list[i].T).T + t
                rmsd = calculate_rmsd(aligned_i, coords_list[j])
            else:
                rmsd = calculate_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd  # Symmetric
    return rmsd_matrix


def calculate_average_coords(
    coords_list: List[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Calculate average (centroid) coordinates from ensemble.
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
    coords_list: List[npt.NDArray[np.float64]],
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Calculate RMSD of each structure to the average structure.
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
