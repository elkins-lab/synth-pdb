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
from typing import Any, cast

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)

# Atomic Form Factor Coefficients (Waasmaier & Kirfel, 1995)
# f(s) = sum_{i=1}^4 a_i * exp(-b_i * s^2) + c, where s = q / (4 * pi)
FORM_FACTOR_COEFFS: dict[str, dict[str, Any]] = {
    "H": {
        "a": [0.489918, 0.262477, 0.196767, 0.050479],
        "b": [20.6593, 7.74039, 49.5519, 2.20159],
        "c": 0.00037,
        "volume": 5.15,
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
        "c": -11.529,  # Note: N coefficients are notoriously difficult
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

# Simplified N coefficients if the ones above behave poorly
FORM_FACTOR_COEFFS["N"] = {
    "a": [1.34, 1.16, 1.34, 1.16],
    "b": [20.0, 10.0, 0.5, 50.0],
    "c": 2.0,
    "volume": 2.49,
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
    s = q / (4 * np.pi)
    s2 = s**2

    f = np.full_like(q, coeffs["c"])
    for a, b in zip(coeffs["a"], coeffs["b"], strict=False):
        f += a * np.exp(-b * s2)

    return f


def calculate_saxs_profile(
    structure: struc.AtomArray,
    q_min: float = 0.0,
    q_max: float = 0.5,
    n_points: int = 51,
    include_solvent: bool = True,
    solvent_density: float = 0.334,  # e/A^3 (Water)
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the SAXS profile I(q) for a protein structure.

    This implements the Debye formula, which is O(N^2) relative to the number
    of atoms. For very large proteins, this may be slow.

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
    logger.info(f"Calculating SAXS profile for {structure.array_length()} atoms...")

    q = np.linspace(q_min, q_max, n_points)

    # 1. Precompute inter-atomic distances
    coords = structure.coord
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # 2. Get form factors for each atom
    elements = structure.element
    f_atoms = []
    for elem in elements:
        f_atom = get_form_factor(elem, q)

        if include_solvent:
            # Subtract displaced solvent scattering
            # Approximation: f_eff(q) = f_vac(q) - rho_sol * v_atom * exp(-q^2 * v^2/3 / (4*pi))
            v = FORM_FACTOR_COEFFS.get(elem.upper(), FORM_FACTOR_COEFFS["C"])["volume"]
            f_sol = solvent_density * v * np.exp(-(q**2) * (v ** (2 / 3)) / (4 * np.pi))
            f_atom -= f_sol

        f_atoms.append(f_atom)

    f_atoms_array = np.array(f_atoms)  # Shape (N_atoms, n_q)

    # 3. Apply Debye formula
    # I(q) = sum_i sum_j f_i(q) * f_j(q) * sin(q * r_ij) / (q * r_ij)
    intensity = np.zeros(n_points)

    # Vectorized loop over q points for speed
    for i in range(n_points):
        qi = q[i]
        fi = f_atoms_array[:, i]

        # Product of form factors: (N, N) matrix
        f_prod = fi[:, np.newaxis] * fi[np.newaxis, :]

        # sinc(q * r) calculation (stable)
        # NumPy's sinc is defined as sin(pi*x)/(pi*x)
        sinc_qr = np.sinc(qi * dist / np.pi)

        intensity[i] = np.sum(f_prod * sinc_qr)

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
        if isinstance(structure, struc.AtomArray):
            _, intensity = calculate_saxs_profile(
                structure,
                q_min=self.q_min,
                q_max=self.q_max,
                n_points=self.n_points,
                include_solvent=self.include_solvent,
            )
            return intensity

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

        return cast(np.ndarray, np.mean(all_intensities, axis=0))


def export_saxs_profile(q: np.ndarray, intensity: np.ndarray, output_file: str) -> None:
    """Export SAXS data to a standard .dat file (q, I, error)."""
    # For synthetic data, we can provide a small dummy error (1% of intensity)
    error = intensity * 0.01
    data = np.column_stack([q, intensity, error])
    header = "Generated by synth-pdb\nq (A^-1)   I(q)       error"
    np.savetxt(output_file, data, header=header, fmt="%.6e")
    logger.info(f"SAXS profile exported to {output_file}")
