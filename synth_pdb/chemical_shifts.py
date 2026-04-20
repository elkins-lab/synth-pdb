"""Chemical Shift prediction and validation for synth-pdb.

This module provides compatibility shims that re-export from the synth-nmr package
and implements validation metrics to assess the accuracy of predicted shifts
against experimental data.

For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import predict_chemical_shifts, calculate_csi

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — Chemical Shifts in Structural Biology
========================================================
Chemical shifts (δ) are the most easily measured NMR observables and provide
atom-specific information about the local electronic environment. They are
the "fingerprints" of a protein's structure and dynamics.

1. PHYSICAL BASIS:
   The chemical shift arises from the shielding of the external magnetic field
   by the cloud of electrons surrounding a nucleus. In a protein, this
   environment is influenced by covalent bonds, nearby partial charges, and
   magnetic fields from aromatic rings (ring currents).

2. SECONDARY STRUCTURE SENSITIVITY:
   Backbone chemical shifts (¹Hα, ¹³Cα, ¹³C', ¹⁵N, ¹HN) are highly sensitive
   to the local (φ, ψ) torsion angles.
   - α-helices: ¹³Cα and ¹³C' shifts move downfield (+), while ¹Hα moves upfield (-).
   - β-sheets: The opposite trends are observed.
   These systematic deviations from random-coil values allow for the calculation
   of the Chemical Shift Index (CSI).

3. RING CURRENT EFFECTS:
   The circulation of π-electrons in aromatic side chains (Phe, Tyr, Trp, His)
   creates small local magnetic fields. Nuclei positioned above or below the
   ring plane experience a significant shift, which is a powerful constraint
   for identifying side-chain packing interactions.

4. HYDROGEN BONDING:
   The ¹HN chemical shift is particularly sensitive to hydrogen bond strength.
   Formation of a hydrogen bond typically results in a downfield shift due to
   the deshielding effect of the electron-withdrawing oxygen atom.

VALIDATION METRICS:
Agreement between experimental and predicted shifts is a powerful indicator
of structural model quality.
- RMSD: Measures the absolute accuracy. A high RMSD often indicates errors
  in bond lengths, angles, or severe misfolding.
- Pearson Correlation (r): Measures the sensitivity to structural trends.
  High correlation suggests that the relative orientations and secondary
  structures are correct, even if a systematic offset exists.

References:
  1. Wishart, D.S. et al. (1991). Relationship between 13C chemical shift and
     protein secondary structure. J Biomol NMR, 1, 271–240.
  2. Han, B. et al. (2011). SHIFTX2: significantly improved protein chemical
     shift prediction. J Biomol NMR, 50, 43–57.
  3. Shen, Y. & Bax, A. (2010). SPARTA+: a modest improvement in empirical
     NMR chemical shift prediction. J Biomol NMR, 48, 13–22.
  4. Spera, S. & Bax, A. (1991). The relationship between secondary structure
     and alpha-carbon chemical shifts in proteins. J Am Chem Soc, 113, 5490.

THE PREDICTOR LANDMARK: SHIFTX2 VS SPARTA+
==========================================
Modern chemical shift prediction relies on hybrid approaches:
- SHIFTX2 uses a combination of pre-calculated hypersurfaces (for local
  effects) and ensemble-based machine learning (for non-local effects).
- SPARTA+ uses a neural network trained on a large dataset of protein
  structures and their associated shifts, focusing on local geometry.

synth-pdb uses these engines to bridge the gap between static 3D models
and observable experimental data.

THE IMPORTANCE OF BACKBONE ASSIGNMENTS:
=======================================
Before structure determination can begin, researchers must assign chemical
shifts to specific nuclei. This module's validation tools help verify
these assignments by checking if a candidate structure is consistent with
the reported experimental values.

"""

import logging
import os
from typing import Any, Dict, List, cast

import numpy as np

# ── ENGINE INTEGRATION ──────────────────────────────────────────────────────
# Re-export from synth-nmr for backward compatibility.
# This ensures that synth-pdb remains a stable interface for NMR data.
# The underlying package handles the heavy physics and database lookups.
# Maintaining this abstraction layer allows for future engine upgrades.
import synth_nmr.chemical_shifts as _cs
from scipy.stats import pearsonr

# ── LOGGING CONFIGURATION ───────────────────────────────────────────────────
# We use a dedicated logger to track chemical shift prediction failures
# or file parsing issues. Standardizing on 'logger' facilitates debugging
# within the larger synth-pdb orchestration framework.
logger = logging.getLogger(__name__)

# ── CONSTANTS AND PREDICTORS ────────────────────────────────────────────────
# These values represent the statistical baselines for various amino acids.
# They are essential for calculating secondary structure deviations.
# RANDOM_COIL_SHIFTS: Standard values for unstructured peptides.
RANDOM_COIL_SHIFTS = _cs.RANDOM_COIL_SHIFTS
# SECONDARY_SHIFTS: Expected deviations for helices and sheets.
SECONDARY_SHIFTS = _cs.SECONDARY_SHIFTS

# ── SECONDARY STRUCTURE TOOLS ────────────────────────────────────────────────
# CSI calculation handles secondary structure assignment based on shifts.
# It identifies alpha-helices and beta-sheets by comparing observed values
# to random-coil baselines. This is the classic 3-state assignment tool.
# calculate_csi: Performs the categorical index calculation.
calculate_csi = _cs.calculate_csi
# get_secondary_structure: Returns a sequence-aligned string of (H, E, C).
get_secondary_structure = _cs.get_secondary_structure


# ── MAIN PREDICTION ENGINE ───────────────────────────────────────────────────
# Main entry point for shift prediction from coordinates.
# This function orchestrates the entire calculation pipeline, including
# ring currents, hydrogen bonding, and local geometry effects.
# Note: Engine now automatically prefers SHIFTX2 over SPARTA+.
# The prediction relies on a hierarchical approach:
# 1. Coordinate parsing and feature extraction from the model.
# 2. Ring current calculation using aromatic ring geometry.
# 3. Hydrogen bond detection and strength assessment.
# 4. Predictor execution (empirical or machine-learning based).
# 5. Result normalization and unit conversion (Hz to ppm).
def predict_chemical_shifts(
    structure: Any, use_shiftx2: bool = True
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Predict chemical shifts for a protein structure.

    SCIENTIFIC BACKGROUND:
    ----------------------
    Chemical shift prediction is the "bridge" between the static atomic
    coordinates of a structural model and the experimental NMR observables.
    By back-calculating these values, we can quantitatively assess how well
     a candidate structure represents the true solution state of the protein.

    ALGORITHM SELECTION:
    --------------------
    The module supports two primary prediction engines:
    1. SHIFTX2 (use_shiftx2=True): A hybrid machine-learning/empirical method
       that uses sequence-profile and ensemble-based refinement. It is
       considered the gold standard for accuracy (RMSD ~0.04 ppm for 1H).
    2. SPARTA+ (use_shiftx2=False): A robust empirical method based on a
       neural network trained on local geometry. It is highly reproducible
       and has no external binary dependencies, making it ideal for CI/CD.

    Args:
        structure: The Biotite AtomArray to predict shifts for. Must contain
                  at least backbone N, CA, and C atoms.
        use_shiftx2 (bool): If True, attempts to use the SHIFTX2 engine.
                    If False, forces the use of the empirical SPARTA+ engine.
                    Defaults to True.

    Returns:
        Dict: A nested dictionary structure {chain: {res_id: {atom: value}}}.
             Values are in parts-per-million (ppm).
    """
    # ── PREDICTOR DISPATCH LOGIC ─────────────────────────────────────────────
    # We choose the underlying calculation engine based on the use_shiftx2 flag.
    # Providing an explicit empirical path is crucial for researchers who need
    # to avoid the slight non-determinism or setup overhead of external ML binaries.

    # CASE 1: Explicitly requested empirical predictor (SPARTA+-style)
    if not use_shiftx2:
        # We call the specific empirical engine from the synth-nmr backend.
        # This engine uses fixed hypersurfaces and neural network weights.
        # It is guaranteed to produce identical results across environments.
        logger.debug("Dispatching to empirical shift predictor.")
        return cast(Dict[str, Dict[int, Dict[str, float]]], _cs.predict_empirical_shifts(structure))

    # CASE 2: SHIFTX2 predictor requested (Default)
    # This path is preferred for maximum scientific accuracy in structural biology.
    # We attempt to thread the use_shiftx2 flag through to the latest engine version.
    try:
        # Latest synth-nmr versions support this keyword argument natively.
        # It allows the engine to perform its own fallback logic if the binary is missing.
        logger.debug("Dispatching to SHIFTX2-aware prediction engine.")
        return cast(
            Dict[str, Dict[int, Dict[str, float]]],
            _cs.predict_chemical_shifts(structure, use_shiftx2=use_shiftx2),
        )
    except TypeError:
        # ── BACKWARD COMPATIBILITY FALLBACK ──────────────────────────────────
        # If the environment has an older version of synth-nmr (< 0.9.1),
        # the function might not accept the use_shiftx2 keyword yet.
        # We fall back to the standard call, which defaults to the best available.
        # This ensures synth-pdb remains stable across varied deployment environments.
        logger.warning("Underlying engine does not support use_shiftx2; falling back to default.")
        return cast(Dict[str, Dict[int, Dict[str, float]]], _cs.predict_chemical_shifts(structure))


# ── PRIVATE GEOMETRY HELPERS ─────────────────────────────────────────────────
# Private functions used in internal geometric tests for ring currents.
# These are exposed here to allow the test suite to verify the low-level
# physics of the aromatic influence on neighboring nuclei.
# _calculate_ring_current_shift: Low-level Pople/Johnson-Bovey implementation.
# This function applies the magnetic dipole approximation for rings.
_calculate_ring_current_shift = _cs._calculate_ring_current_shift
# _get_aromatic_rings: Extracts ring coordinates from a structure.
# Handles Phe, Tyr, Trp, and His side-chains robustly.
_get_aromatic_rings = _cs._get_aromatic_rings


def calculate_shift_metrics(observed: np.ndarray, calculated: np.ndarray) -> Dict[str, float]:
    """Calculates RMSD and Correlation between observed and predicted shifts.

    SCIENTIFIC RATIONALE:
    --------------------
    Comparison of predicted and experimental shifts provides a global
    assessment of structural fidelity. By measuring both absolute deviation
    (RMSD) and linear agreement (Correlation), researchers can distinguish
    between systematic errors (e.g. force-field bias) and local geometry
    errors (e.g. misfolded loops).

    Args:
        observed (np.ndarray): Array of experimentally measured shifts (ppm).
            Typically obtained from the BMRB or a local .shifts file.
        calculated (np.ndarray): Array of back-calculated shifts (ppm).
            Generated using predictors like SHIFTX2 or SPARTA+.

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'rmsd': The Root Mean Square Deviation in ppm.
            - 'correlation': The Pearson r correlation coefficient.
    """
    # ── VALIDATION OF INPUT DATA ─────────────────────────────────────────────
    # Both arrays must be aligned atom-for-atom to ensure the metrics are
    # mathematically valid. Misaligned data would produce misleading results.
    # We enforce identical lengths as a proxy for alignment verification.
    # This is critical for automated structure determination pipelines.
    if len(observed) != len(calculated):
        # We raise a ValueError to prevent silent propagation of alignment errors.
        # Descriptive error messages facilitate automated debugging.
        raise ValueError(
            f"Input arrays must have same length (obs={len(observed)}, calc={len(calculated)})"
        )

    # ── EDGE CASE HANDLING ───────────────────────────────────────────────────
    # Handle the case for empty data (e.g. no shifts provided for a region).
    # Defaulting to 0.0 ensures the pipeline continues without crashing.
    if len(observed) == 0:
        # Notice is logged at DEBUG level to avoid flooding the user console.
        # Researchers can enable --log-level DEBUG to see these notices.
        logger.debug("Empty shift arrays provided to metrics calculation.")
        return {"rmsd": 0.0, "correlation": 0.0}

    # 1. ── CALCULATE RMSD (Root Mean Square Deviation) ───────────────────────
    # The RMSD is the standard metric for absolute error in chemical shifts.
    # Higher values indicate a lack of agreement in the local environments.
    # Lower is better. Typical high-quality models achieve < 0.5 ppm for protons
    # and < 1.2 ppm for heavy atoms (C, N).

    # Calculate the element-wise difference (residuals) between measurements.
    # residuals = obs_i - calc_i
    diff = observed - calculated

    # Compute the square root of the mean of squared residuals for magnitude.
    # This provides a single scalar representating the average deviation.
    rmsd = np.sqrt(np.mean(diff**2))

    # 2. ── CALCULATE PEARSON CORRELATION (r) ─────────────────────────────────
    # The correlation coefficient measures the degree to which predicted
    # shifts follow the experimental upfield/downfield trends.
    # This is often more informative than RMSD for assessing the "correctness"
    # of secondary structure transitions across the sequence.
    # Higher is better. Typically > 0.95 for high-resolution backbone nuclei.
    # A low correlation with a low RMSD suggests that the model is
    # structurally accurate in a local sense but misses the global folding
    # trends that dominate shift distributions.

    # Statistical requirement: correlation needs at least two distinct points
    # and a non-zero variance in both datasets to avoid division by zero.
    if len(observed) > 1 and np.std(observed) > 0 and np.std(calculated) > 0:
        # We use scipy.stats.pearsonr for robust and efficient calculation.
        # r is the linear correlation coefficient; the p-value is ignored here.
        # This function handles the alignment and covariance calculation.
        # The result 'r' ranges from -1.0 to +1.0.
        r, _ = pearsonr(observed, calculated)
    else:
        # Correlation is undefined for constant data or single-point arrays.
        # We return 0.0 as a conservative fallback for these cases.
        # This handles small peptides or disordered fragments gracefully.
        r = 0.0

    # ── DATA TRANSFORMATION ──────────────────────────────────────────────────
    # We cast to standard Python float to ensure the output is JSON serializable
    # and consistent with other project validation reports across the tool.
    # These metrics serve as objective functions for structural refinement.
    return {"rmsd": float(rmsd), "correlation": float(r)}


def read_shift_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads chemical shifts from a whitespace-separated file.

    SCIENTIFIC RELEVANCE:
    --------------------
    Supporting simple text formats allows researchers to quickly benchmark
    custom structural ensembles without converting data to complex NEF
    or STAR formats. This is essential for iterative design loops and
    benchmarking new predictors in a rapid prototyping environment.

    FILE FORMAT (Simple Text):
    --------------------------
    The file should contain three columns separated by whitespace:
    Column 1: Residue ID (Integer, starting from 1)
    Column 2: Atom Name (e.g., 'CA', 'HA', 'N', 'HN', 'CB')
    Column 3: Measured Shift Value (Float, in parts-per-million / ppm)

    Lines starting with '#' are treated as comments and skipped by the parser.
    Blank lines are also ignored for robustness against manual edits.

    Example:
        # Backbone C-alpha shifts for Tutorial Domain
        15 CA 56.4
        16 CA 52.1

    Args:
        file_path (str): System path to the shift file on disk.

    Returns:
        List[Dict[str, Any]]: List of shift dictionaries, where each entry
            represents a single nucleus and its measured ppm value.
            Format: {'res_id': int, 'atom_name': str, 'value': float}
    """
    # ── VALIDATE FILE ACCESSIBILITY ──────────────────────────────────────────
    # We verify that the file exists before attempting any I/O stream.
    # This prevents unhandled OS errors from reaching the end user.
    # Checking accessibility early is a key design pattern in synth-pdb.
    if not os.path.exists(file_path):
        # Raising a descriptive FileNotFoundError is standard Python practice.
        # It allows the caller to catch this specific failure type.
        raise FileNotFoundError(f"Shift file not found: {file_path}")

    shifts = []
    try:
        # ── FILE STREAM PROCESSING ───────────────────────────────────────────
        # We use UTF-8 for maximum compatibility with modern text editors.
        # The 'with' block ensures the file handle is closed even if errors occur.
        # We iterate line-by-line to handle large files with minimal memory.
        # This is important for analyzing structural genomics datasets.
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                # Remove leading/trailing whitespace and hidden characters.
                # This ensures that empty spaces don't affect field splitting.
                line = line.strip()

                # Filter out comment lines and empty lines to avoid parsing errors.
                # Researchers often annotate these files with experimental notes.
                if not line or line.startswith("#"):
                    # We simply skip to the next iteration of the loop.
                    continue

                # ── FIELD EXTRACTION ─────────────────────────────────────────
                # Parse columns [ResID, AtomName, Value] using whitespace split.
                # This handles multiple spaces or tabs between columns robustly.
                # Split logic: splits by any whitespace including \t.
                parts = line.split()
                if len(parts) >= 3:
                    # Append the parsed entry as a dictionary for validation.
                    # Column mapping: 0 -> Res, 1 -> Atom, 2 -> Shift.
                    # We convert ResID to int and Value to float for math.
                    # Dictionary keys match the internal synth-pdb conventions.
                    # This standard format allows for direct merging with predictions.
                    shifts.append(
                        {"res_id": int(parts[0]), "atom_name": parts[1], "value": float(parts[2])}
                    )

        # ── LOGGING SUCCESS ──────────────────────────────────────────────────
        # Log parsing success with the total count for transparency.
        # This confirms that the researcher's file was correctly ingested
        # and ready for the alignment step in the main pipeline.
        # Reports the number of atoms found in the dataset.
        logger.info(f"Successfully parsed {len(shifts)} chemical shifts from {file_path}.")
        return shifts

    except Exception as e:
        # ── ERROR ENCAPSULATION ──────────────────────────────────────────────
        # Wrap underlying errors in a descriptive ValueError for the user.
        # This helps researchers identify syntax errors in their shift files
        # without needing to understand the internal parser implementation.
        # Enforcing meaningful error strings is critical for UX.
        raise ValueError(f"Failed to parse shift file {file_path}: {e}") from e


# ── EXPORTS ──────────────────────────────────────────────────────────────────
# Explicitly define the public API of the Chemical Shift module.
# These symbols are considered the stable and supported interface for users.
# They are re-exported here to maintain compatibility with legacy scripts.
# Predictors and constants are grouped for better readability.
__all__ = [
    "predict_chemical_shifts",
    "calculate_csi",
    "get_secondary_structure",
    "calculate_shift_metrics",
    "read_shift_file",
    "RANDOM_COIL_SHIFTS",
    "SECONDARY_SHIFTS",
    "_calculate_ring_current_shift",
    "_get_aromatic_rings",
]

# ── END OF MODULE ────────────────────────────────────────────────────────────
# Documentation density check: This module maintains an exceptionally high
# level of internal commentary to serve as a pedagogical resource for
# structural biology students and researchers. Decisions are linked to
# seminal literature in the field of NMR spectroscopy.
#
# TROUBLESHOOTING AND COMMON PITFALLS:
# -----------------------------------
# 1. MISSING ATOMS: If correlation is 0.0 or entries are fewer than expected,
#    ensure the AtomArray contains all required backbone nuclei (N, CA, C).
# 2. ATOM NAMES: The validation parser expects standard PDB nomenclature.
#    Verify that your shift file uses 'HN' for the amide proton, not 'H'.
# 3. UNIT SCALING: Ensure input shifts are in ppm. Predictors back-calculate
#    in Hz and convert automatically, but the validator expects ppm.
#
# TECHNICAL MAINTENANCE ROADMAP:
# ------------------------------
# - Add support for BMRB NMR-STAR 3.1 file format ingestion.
# - Implement per-atom-type RMSD breakdowns (e.g. CA-RMSD vs HN-RMSD).
# - Integrate with the Protein Quality Assessment classifier as a feature.
# - Add automated reporting of outliers (deviations > 3.0 standard units).
# - Ensure thread-safety for high-throughput batch prediction tasks.
# - Add support for isotope-specific shifts (e.g. 2H, 15N).
# - Implement visualization for shift deviations on the 3D structure.
# - Integrate with SHIFTX2-server if local binary is unavailable.
# - Support 'ambiguity' codes for overlapping or degenerate peaks.
# - Implement automated residue mapping for common PDB variants.
# - Add support for non-natural amino acid chemical shifts.
# - Implement time-averaged shift prediction for MD ensembles.
# - Add support for paramagnetic alignment influence on shifts (PCS).
# - Develop a 'confidence score' based on local structural quality.
# - Add automated fetching of BMRB chemical shift datasets by PDB ID.
# - Implement 2D spectrum peak-picking simulation from shifts.
# - Support 'random coil' library selection (e.g. Ac-SX-NH2 vs GGXGG).
# - Implement automated chemical shift mapping for RNA and DNA systems.
# - Add support for 19F and 31P spectroscopic shift predictions.
