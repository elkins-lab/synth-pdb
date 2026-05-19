"""
# EDUCATIONAL OVERVIEW - SAXS Curve Simulation:
# ---------------------------------------------
# Small-Angle X-ray Scattering (SAXS) is a fundamental technique for studying
# protein structure and dynamics in solution. This module computes synthetic
# scattering curves (I(q) vs q) from atomic coordinates.
#
# SCIENTIFIC PRINCIPLES:
# ----------------------
# 1. The Debye Formula: The scattering intensity I(q) is computed by summing the
#    interference between all pairs of atoms in the molecule.
#    I(q) = sum_i sum_j f_i(q) f_j(q) * sin(q * r_ij) / (q * r_ij)
#    where q is the scattering vector magnitude and r_ij is the distance between
#    atoms i and j.
#
# 2. Atomic Form Factors: Atoms of different elements scatter X-rays with
#    different efficiencies. We use q-dependent form factors approximated by
#    a sum of Gaussians.
#
# 3. Solvent Contrast: In SAXS, we measure the "excess" scattering of the protein
#    relative to the solvent. We subtract the scattering contribution of the
#    displaced solvent volume for each atom.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Atomic Form Factor Coefficients (Waasmaier & Kirfel, 1995)
# f(s) = sum_{i=1}^4 a_i * exp(-b_i * s^2) + c, where s = q / (4 * pi)
# Volumes (A^3) derived from CRYSOL (Svergun et al., 1995)
FORM_FACTOR_COEFFS: dict[str, dict[str, Any]] = {
    "H": {
        "a": [0.489918, 0.262477, 0.196767, 0.050479],
        "b": [20.6593, 7.74039, 49.5519, 2.20159],
        "c": 0.00037,
        "volume": 0.0,  # Hydrogens typically have negligible displaced volume in this model
    },
    "C": {
        "a": [2.31, 1.02, 1.5886, 0.865],
        "b": [20.8439, 10.2075, 0.5687, 51.6512],
        "c": 0.2156,
        "volume": 16.44,
    },
    "N": {
        "a": [12.2126, 3.1322, 2.0125, 1.1663],
        "b": [0.0057, 9.8933, 28.9974, 0.5826],
        "c": -11.529,
        "volume": 2.49,
    },
    "O": {
        "a": [3.0485, 2.2868, 1.5463, 0.867],
        "b": [13.2771, 5.7011, 0.3239, 32.908],
        "c": 0.2508,
        "volume": 9.13,
    },
    "S": {
        "a": [6.9053, 5.2034, 1.4379, 1.5861],
        "b": [1.4679, 22.2151, 0.2536, 56.172],
        "c": 0.8669,
        "volume": 19.86,
    },
    "P": {
        "a": [6.4345, 4.1791, 1.782, 1.4908],
        "b": [1.9067, 27.157, 0.526, 68.1641],
        "c": 1.1149,
        "volume": 24.4,
    },
}


def get_form_factor(element: str, q: np.ndarray) -> np.ndarray:
    """Compute the q-dependent form factor for a given element.

    Args:
        element: Element symbol (e.g. 'C', 'N', 'O').
        q: 1D array of scattering vector magnitudes (Angstroms^-1).

    Returns:
        np.ndarray: Form factor values for each q.
    """
    element = element.upper()
    if element not in FORM_FACTOR_COEFFS:
        # Fallback to Carbon if element unknown
        element = "C"

    coeffs = FORM_FACTOR_COEFFS[element]
    s2 = (q / (4 * np.pi)) ** 2

    f = np.full_like(q, coeffs["c"])
    for a, b in zip(coeffs["a"], coeffs["b"], strict=False):
        f += a * np.exp(-b * s2)

    return f


def calculate_radius_of_gyration(structure: struc.AtomArray) -> float:
    """Calculate the Radius of Gyration (Rg) of a structure.

    Args:
        structure: Biotite AtomArray.

    Returns:
        float: Radius of gyration in Angstroms.
    """
    return float(struc.gyration_radius(structure))


def calculate_saxs_profile(
    structure: struc.AtomArray,
    q_min: float = 0.0,
    q_max: float = 0.5,
    n_points: int = 51,
    include_solvent: bool = True,
    solvent_density: float = 0.334,  # e/A^3 (Water)
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the SAXS profile I(q) for a protein structure.

    This implements the Debye formula with O(N^2) complexity.

    Args:
        structure: Biotite AtomArray (full atom recommended).
        q_min: Minimum q value (default 0.0).
        q_max: Maximum q value (default 0.5).
        n_points: Number of q points.
        include_solvent: If True, subtracts displaced solvent volume.
        solvent_density: Electron density of the solvent.

    Returns:
        Tuple of (q_values, intensity_values).
    """
    n_atoms = structure.array_length()
    logger.info(f"Calculating SAXS profile for {n_atoms} atoms...")

    q = np.linspace(q_min, q_max, n_points)

    # 1. Precompute inter-atomic distances (N x N matrix)
    coords = structure.coord
    if coords.ndim == 3:
        # If passed an AtomArrayStack with 1 model, flatten to 2D
        coords = coords[0]

    # Use scipy for efficient distance calculation
    dist = cdist(coords, coords)

    # 2. Vectorized form factor calculation
    elements = structure.element
    unique_elements = np.unique(elements)
    f_atoms_array = np.zeros((n_atoms, n_points))

    for elem in unique_elements:
        mask = elements == elem
        f_atom = get_form_factor(elem, q)

        if include_solvent:
            # Solvent displacement: f_eff = f_vac - rho_sol * V * exp(-q^2 * R^2 / 10)
            # R is the effective atomic radius: R = (3V / 4pi)^(1/3)
            # We use 1/10 in the exponent for a better sphere approximation than 1/6
            v = FORM_FACTOR_COEFFS.get(elem.upper(), FORM_FACTOR_COEFFS["C"])["volume"]
            decay_rate = ((3 * v) / (4 * np.pi)) ** (2 / 3) / 10.0
            f_sol = solvent_density * v * np.exp(-(q**2) * decay_rate)
            f_atom = f_atom - f_sol

        f_atoms_array[mask] = f_atom

    # 3. Apply Debye formula: I(q) = sum_i sum_j f_i(q) * f_j(q) * sinc(q * r_ij)
    intensity = np.zeros(n_points)

    for i in range(n_points):
        qi = q[i]
        fi = f_atoms_array[:, i]

        if qi < 1e-7:
            # At q=0, sinc(qr) = 1, so I(0) = (sum f_i) ** 2
            intensity[i] = np.sum(fi) ** 2
        else:
            # Use np.sinc for numerical stability
            # Note: np.sinc(x) is sin(pi*x) / (pi*x), so we pass qr/pi
            qr_over_pi = (qi * dist) / np.pi
            sinc_qr = np.sinc(qr_over_pi)

            # Use dot product for faster summation: fi^T * sinc_qr * fi
            intensity[i] = fi @ (sinc_qr @ fi)

    return q, intensity


class SaxsSimulator:
    """Stateful SAXS simulator for ensembles."""

    def __init__(
        self,
        q_min: float = 0.0,
        q_max: float = 0.5,
        n_points: int = 51,
        include_solvent: bool = True,
    ):
        self.q_min = q_min
        self.q_max = q_max
        self.n_points = n_points
        self.include_solvent = include_solvent

    def simulate(self, structure: struc.AtomArray | struc.AtomArrayStack) -> np.ndarray:
        """Computes the averaged SAXS profile for a structure or ensemble."""
        if hasattr(structure, "stack_depth") and structure.stack_depth() > 0:
            # For ensembles, average the intensities
            all_intensities = []
            for i in range(structure.stack_depth()):
                _, intensity = calculate_saxs_profile(
                    structure[i],
                    q_min=self.q_min,
                    q_max=self.q_max,
                    n_points=self.n_points,
                    include_solvent=self.include_solvent,
                )
                all_intensities.append(intensity)

            if all_intensities:
                return cast(np.ndarray, np.mean(all_intensities, axis=0))
            return np.zeros(self.n_points)

        if isinstance(structure, struc.AtomArrayStack) and structure.stack_depth() == 0:
            logger.warning("Attempted to simulate SAXS on an empty ensemble.")
            return np.zeros(self.n_points)

        # Single structure
        _, intensity = calculate_saxs_profile(
            structure,  # type: ignore[arg-type]
            q_min=self.q_min,
            q_max=self.q_max,
            n_points=self.n_points,
            include_solvent=self.include_solvent,
        )
        return intensity


def export_saxs_profile(q: np.ndarray, intensity: np.ndarray, output_file: str) -> None:
    """Export SAXS data to a standard .dat file (q, I, error)."""
    # For synthetic data, we can provide a small dummy error (1% of intensity)
    error = intensity * 0.01
    data = np.column_stack([q, intensity, error])
    header = "Generated by synth-pdb\nq (A^-1)   I(q)       error"
    np.savetxt(output_file, data, header=header, fmt="%.6e")
    logger.info(f"SAXS profile exported to {output_file}")
