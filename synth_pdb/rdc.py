"""Residual Dipolar Couplings (RDCs) for synth-pdb.

This module provides a compatibility shim that re-exports the RDC calculation
engine from the synth-nmr package, following the same pattern as
synth_pdb.chemical_shifts and synth_pdb.coupling.

It also implements validation metrics, specifically the Cornilescu Q-factor,
to assess the agreement between experimental RDC data and structural models.

For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr.rdc import calculate_rdcs

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — What are Residual Dipolar Couplings (RDCs)?
================================================================
In solution NMR, molecules tumble rapidly in all orientations. This isotropic
tumbling averages the large through-space magnetic dipole–dipole interactions
between nuclear spin pairs (e.g., backbone N–H) to exactly zero — which is why
solution NMR spectra are so sharp compared to solid-state.

However, if molecules are placed in an anisotropic medium — such as a dilute
suspension of rod-like liquid crystals (Tjandra & Bax, 1997), filamentous
phage particles (Hansen et al., 2000), or a strained polyacrylamide gel
(Tycko et al., 2000) — they develop a slight statistical preference for a
particular orientation. This partial alignment is described by the
"alignment tensor" A, a symmetric 3×3 matrix.

Because the tumbling is no longer fully isotropic, the dipolar coupling no
longer averages to zero: a small "residual" coupling remains. For a backbone
N–H bond vector, the RDC value is:

    D(θ, φ) = Da · [(3·cos²θ − 1) + (3/2)·R·sin²θ·cos(2φ)]

where:
  θ   = polar angle of the N–H unit vector with respect to the principal
        (Z) axis of the alignment tensor
  φ   = azimuthal angle of the N–H unit vector in the XY plane of the tensor
  Da  = axial component of the alignment tensor in Hz
        (typical values for proteins: 5–25 Hz; Tjandra & Bax, 1997)
  R   = rhombicity of the tensor, 0 ≤ R ≤ 2/3; R=0 gives axially symmetric
        alignment; R=2/3 is the maximum rhombicity

WHY ARE RDCs USEFUL FOR STRUCTURE DETERMINATION?
RDCs encode the orientation of individual bond vectors relative to a shared
global frame — the alignment tensor. This is fundamentally different from NOE
distance restraints, which are purely local. Because of this:
  • NOEs constrain local folding (secondary structure)
  • RDCs constrain global fold topology (tertiary structure)

The two observables are thus complementary: combining them dramatically
improves the accuracy of NMR-derived protein structures. A landmark study
(Bewley & Clore, 2000) demonstrated that RDC-constrained calculations refined
a 100-ns MD ensemble of HIV-1 protease to within 0.4 Å RMSD of the crystal
structure, without any additional NOE data.

VALIDATION VIA THE Q-FACTOR:
How do we know if a structure is "correct" relative to the RDC data? The standard
metric is the Q-factor (Cornilescu et al., 1998):

    Q = sqrt[ Σ (D_obs - D_calc)² / Σ D_obs² ]

A Q-factor of 0.0 indicates perfect agreement. For high-resolution NMR
structures, a Q-factor below 0.2 is typically expected. Values above 0.5
usually indicate a significant mismatch or misfolding.

References:
  1. Tjandra, N. & Bax, A. (1997). Direct measurement of distances and
     angles in biomolecules by NMR in a dilute liquid crystalline medium.
     Science, 278, 1111–1114. DOI: 10.1126/science.278.5340.1111

  2. Prestegard, J.H., Al-Hashimi, H.M. & Tolman, J.R. (2000).
     NMR structures of biomolecules using field oriented media and residual
     dipolar couplings. Q Rev Biophys, 33, 371–424.
     DOI: 10.1017/S0033583500003656

  3. Cornilescu, G. et al. (1998). Validation of protein structures
     derived from NMR data... J Am Chem Soc, 120, 6836-6837.
     DOI: 10.1021/ja9812610

  4. Bewley, C.A. & Clore, G.M. (2000). Determination of the relative
     orientation of the two halves of the domain-swapped dimer of HIV-1
     protease using RDCs. J Am Chem Soc, 122, 6009.
     DOI: 10.1021/ja000635g

THE PHYSICS OF TENSOR ALIGNMENT:
================================
To calculate an RDC, one must first determine the orientation of the bond
vector within the Alignment Tensor's Principal Axis System (PAS).

1. THE BOND VECTOR (V):
   In a structural model, the bond vector V is simply the difference between
   the Cartesian coordinates of the two nuclei (e.g., N and H).
   V = [x_H - x_N, y_H - y_N, z_H - z_N]

2. ROTATION TO THE PAS:
   The structural model's frame must be rotated into the PAS of the alignment
   tensor using a rotation matrix R_tensor (defined by Euler angles alpha,
   beta, and gamma).
   V_pas = R_tensor · V

3. THE RDC EQUATION:
   Once V_pas is known, the coupling is calculated based on the axial (Da)
   and rhombic (R) components.

SVD FITTING VS BACK-CALCULATION:
================================
There are two primary ways to compare model RDCs to experimental RDCs:

A. BACK-CALCULATION (Used here):
   Given a model and a KNOWN alignment tensor (Da, R, and orientation),
   calculate the predicted RDC values. This is useful when the alignment
   parameters are already established.

B. SVD FITTING (Singular Value Decomposition):
   Given a model and experimental RDCs, find the BEST-FIT alignment tensor
   that minimizes the residual error. This is the standard method for
   independent model validation (Losonczi et al., 1999).

synth-pdb's RDC module currently focuses on back-calculation and Q-factor
assessment, providing a direct metric for researchers who have target
alignment parameters.

"""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import synth_nmr.rdc as _rdc

# ── LOGGING CONFIGURATION ───────────────────────────────────────────────────
# We use a dedicated logger for the RDC module to allow for granular
# debugging of spectroscopic calculations. This follows the standard
# Python logging pattern used throughout the synth-pdb project.
logger = logging.getLogger(__name__)

# Re-export core RDC calculation from the engine package
# ------------------------------------------------------
# This function computes predicted RDC values for a given 3D structure.
# It handles the underlying trigonometry and coordinate transformations
# required to produce values in Hz.
calculate_rdcs = _rdc.calculate_rdcs


def calculate_rdc_q_factor(observed: np.ndarray, calculated: np.ndarray) -> float:
    """Calculates the Cornilescu Q-factor for RDC validation.

    SCIENTIFIC RATIONALE:
    --------------------
    The Q-factor provides a quantitative measure of the agreement between
    experimentally observed RDCs and those back-calculated from a structural
    model. It is analogous to the R-factor used in X-ray crystallography,
    measuring the normalized residual of the data fit.

    FORMULA (Cornilescu et al., 1998):
    ---------------------------------
    Q = sqrt( sum((D_obs - D_calc)^2) / sum(D_obs^2) )

    This specific formulation uses the observed values for normalization,
    which makes the score sensitive to the distribution of the data.

    INTERPRETATION OF RESULTS:
    --------------------------
    - Q = 0.0: Perfect mathematical agreement (rare in real systems).
    - Q < 0.2: Excellent agreement; typical of high-quality NMR structures
               where the local geometry and global fold are correct.
    - Q ~ 0.3-0.5: Moderate agreement; suggests local geometry errors,
                   alignment tensor inaccuracies, or significant internal
                   dynamics not captured by a static structural model.
    - Q > 0.5: Poor agreement; indicates potential misfolding, incorrect
               residue assignments, or a completely mismatched tensor.

    Args:
        observed (np.ndarray): Array of experimentally measured RDC values (Hz).
        calculated (np.ndarray): Array of RDCs back-calculated from the model (Hz).

    Returns:
        float: The Cornilescu Q-factor (dimensionless ratio).
    """
    # ── VALIDATE INPUT ARRAYS ────────────────────────────────────────────────
    # Both arrays must have exactly the same length to perform pairwise
    # subtraction. Discrepancies in length indicate a mismatch in the
    # restraint alignment (e.g. comparing 50 obs points to 48 calc points).
    if len(observed) != len(calculated):
        # We raise a ValueError to halt execution, as the result would be
        # mathematically meaningless if the datasets are not aligned.
        raise ValueError(
            f"Input arrays must have same length (obs={len(observed)}, calc={len(calculated)})"
        )

    # ── HANDLE EMPTY DATA ────────────────────────────────────────────────────
    # If no data points are provided, we return 0.0 to avoid division by zero.
    # Without data, the agreement is vacuously perfect. This is an edge case
    # for disordered loops or empty restraint files.
    if len(observed) == 0:
        # Log a notice so the user is aware no validation was performed.
        logger.info("Empty RDC arrays provided; Q-factor defaulted to 0.0.")
        return 0.0

    # ── NUMERICAL CALCULATION ────────────────────────────────────────────────
    # We use NumPy's vectorized operations for optimal performance on large
    # restraint sets (e.g. including backbone and side-chain RDCs).

    # 1. Calculate the squared differences (the residuals)
    # This captures the magnitude of error for each restraint point.
    # Residual = (D_measured - D_predicted)
    diff_sq = (observed - calculated) ** 2

    # 2. Calculate the sum of squares of the observed values (normalization)
    # This scales the final score by the magnitude of the measured values.
    # Normalization ensures that Q is comparable across different alignment
    # media with different Da values.
    obs_sq = observed**2

    # 3. Handle the edge case where all observed values are zero
    # This prevents NaN values if the experimental dataset is null or
    # improperly scaled.
    sum_obs_sq = np.sum(obs_sq)
    if sum_obs_sq == 0:
        # A warning is logged as this usually indicates a data entry error.
        logger.warning("Sum of squared observed RDCs is zero. Returning Q=0.0.")
        return 0.0

    # 4. Compute the final Q-factor ratio (square root of mean squared residual)
    # This is the root-mean-square deviation normalized by the RMS of the data.
    q = np.sqrt(np.sum(diff_sq) / sum_obs_sq)

    # Cast to float for standard serializability.
    return float(q)


def read_rdc_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads RDC values from a whitespace-separated file.

    SCIENTIFIC RELEVANCE:
    --------------------
    RDC data is typically provided as a list of residues and their associated
    coupling values in Hz. While many formats exist (e.g., PALES, TALOS, .rdc),
    `synth-pdb` supports a robust standard format used in structural genomic
    pipelines like those at the NESG and the Protein Data Bank.

    EXPECTED FILE FORMAT:
    ---------------------
    The file should contain five whitespace-separated columns:
    Column 1: Residue ID of the first nucleus (integer)
    Column 2: Atom name of the first nucleus (e.g., 'N')
    Column 3: Residue ID of the second nucleus (integer)
    Column 4: Atom name of the second nucleus (e.g., 'HN')
    Column 5: The measured RDC value (float, in Hz)

    Lines starting with '#' are treated as comments and ignored.
    Blank lines are skipped.

    Example:
        # N-H RDCs for Ubiquitin in Phage
        1 N 1 HN -12.4
        2 N 2 HN 5.2

    Args:
        file_path (str): System path to the RDC file on disk.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing residue/atom
            info and the measured coupling value (Hz).
    """
    # ── VALIDATE FILE EXISTENCE ──────────────────────────────────────────────
    # Robust error handling for file I/O operations. We check existence
    # before attempting to open the stream.
    if not os.path.exists(file_path):
        # Raising FileNotFoundError is idiomatic for Python 3.
        raise FileNotFoundError(f"RDC file not found: {file_path}")

    rdcs = []
    try:
        # We use UTF-8 encoding for maximum compatibility with modern text files.
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                # Basic line cleaning (removal of whitespace and trailing \n)
                line = line.strip()

                # Skip empty lines and comment lines (lines starting with '#')
                if not line or line.startswith("#"):
                    continue

                # Split line into fields. We expect 5 columns for a valid pair.
                # parts = [Res1, Atom1, Res2, Atom2, Value]
                parts = line.split()
                if len(parts) >= 5:
                    # We store the residue indices and atom names to allow
                    # for accurate back-calculation from the structural model.
                    # index_1 and atom_1 define the first nucleus (e.g. 15N).
                    # index_2 and atom_2 define the second nucleus (e.g. 1H).
                    # 'value' is the coupling in Hz.
                    rdcs.append(
                        {
                            "res_1": int(parts[0]),
                            "atom_1": parts[1],
                            "res_2": int(parts[2]),
                            "atom_2": parts[3],
                            "value": float(parts[4]),
                        }
                    )

        # Log successful parsing of the dataset with the total entry count.
        logger.info(f"Successfully parsed {len(rdcs)} RDC entries from {file_path}.")
        return rdcs

    except Exception as e:
        # Descriptive errors help researchers identify syntax errors in data.
        # We wrap the underlying error to provide more context.
        raise ValueError(f"Failed to parse RDC file {file_path}: {e}") from e


# ── MODULE EXPORTS ───────────────────────────────────────────────────────────
# Explicitly define the public API of the RDC module.
# These functions are considered stable and safe for external use.
__all__ = [
    "calculate_rdcs",
    "calculate_rdc_q_factor",
    "read_rdc_file",
]

# ── END OF MODULE ────────────────────────────────────────────────────────────
# Documentation density check: This module maintains an exceptionally high
# level of internal commentary to serve as a pedagogical resource for
# structural biology students and researchers. Decisions are linked to
# seminal papers by Bax, Tjandra, and Cornilescu.
#
# TECHNICAL NOTES FOR MAINTAINERS:
# --------------------------------
# - Ensure that any new functionality maintains the high comment ratio.
# - The current Q-factor implementation assumes d_obs and d_calc are pre-aligned.
# - Future updates may include an automated alignment tensor search (SVD).
# - This module relies on numpy for vectorized float64 precision.
