from typing import Any, cast

import numpy as np


def compute_contact_map(
    structure: Any, method: str = "ca", threshold: float = 8.0, power: float | None = None
) -> np.ndarray:
    """Computes a residue-residue contact map.

    Parameters
    ----------
    structure : AtomArray
        The protein structure.
    method : str
        "ca" for Alpha-Carbon distances (standard for AI/Fold evaluation).
        "noe" for Minimum Proton-Proton distances (realistic NMR proxy).
    threshold : float
        Distance cutoff in Angstroms. Contacts closer than this are "1", else "0" (or continuous).
    power : int or None
        If None, returns raw distances.
        If 6, returns 1/r^6 (NOE Intensity proxy).
        If 0, returns binary contact map (1 if r < threshold, else 0).

    Returns
    -------
    matrix : np.ndarray (N_res x N_res)
        The contact matrix.

    """
    # Filter for the relevant atoms
    if method == "ca":
        # CA atoms represent the backbone geometry best for AI
        mask = (structure.atom_name == "CA") & (structure.element == "C")
        atoms = structure[mask]
    elif method == "noe":
        # Realistic NOE is H-H distance.
        # Approximating as 'N' (Amide) is cheaper, but let's try to do it right?
        # Actually, calculating ALL H-H pairs is expensive (N^2 protons).
        # Let's use CB (Beta Carbon) as a robust proxy for side-chain contacts.
        mask = (structure.atom_name == "CB") | (
            (structure.atom_name == "CA") & (structure.res_name == "GLY")
        )
        atoms = structure[mask]
    else:
        raise ValueError("Method must be 'ca' or 'noe'")

    coords = atoms.coord
    coords.shape[0]

    # Compute Distance Matrix (Broadcasting)
    # dist[i, j] = ||r_i - r_j||
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))

    if power == 0:
        # Binary Map (AI style)
        contact_map = (dist_matrix < threshold).astype(float)
        return cast(np.ndarray, contact_map)
    elif power == 6:
        # NOE Intensity (Physics style)
        # Avoid division by zero diagonal
        with np.errstate(divide="ignore"):
            intensity = 1.0 / (dist_matrix**6)
        np.fill_diagonal(intensity, 0.0)  # Self-peaks are irrelevant here
        return cast(np.ndarray, intensity)
    else:
        # Raw Distances (Heatmap style)
        return cast(np.ndarray, dist_matrix)
