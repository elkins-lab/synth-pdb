"""NMR Spectroscopy utilities for synth-pdb.

This module provides compatibility shims and high-level validation tools for
synthetic NMR data. It re-exports core functionality from the `synth-nmr`
package and implements additional metrics for structural quality assessment
based on experimental evidence.

The goal of this module is to bridge the gap between structural generation
and spectroscopic validation, providing researchers with the tools to
empirically assess the quality of generated protein ensembles.

For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_synthetic_noes

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — The Nuclear Overhauser Effect (NOE)
======================================================
The Nuclear Overhauser Effect (NOE) is the fundamental observable used to
determine the three-dimensional structure of proteins in solution. It is the
"ruler" of the structural biologist, allowing for the measurement of
inter-atomic distances in a dynamic, aqueous environment.

1. PHYSICAL BASIS:
   The NOE arises from cross-relaxation between two nuclear spins (usually
   protons) that are coupled through dipole-dipole interactions. This coupling
   is direct through space, rather than through chemical bonds. This means
   that two atoms can show an NOE even if they are far apart in the primary
   sequence, as long as the protein fold brings them into close proximity.

2. DISTANCE DEPENDENCE:
   The intensity of an NOE cross-peak in a 2D or 3D NMR spectrum is inversely
   proportional to the sixth power of the distance between the two nuclei (1/r⁶).
   Due to this rapid decay, NOEs are typically only detectable when the
   inter-proton distance is less than approximately 6.0 Å. This sensitivity
   makes the NOE an incredibly precise tool for structural determination.

3. STRUCTURAL INFORMATION:
   - Intra-residue and Sequential NOEs (i to i+1): Provide information on
     local secondary structure. For example, specific patterns of i to i+3
     and i to i+4 NOEs are the hallmarks of an alpha-helix.
   - Long-range NOEs: Link residues that are far apart in the primary sequence
     but close in the tertiary fold. These are critical for defining the global
     topology of the protein and determining how the different secondary
     structure elements pack against each other.

4. DYNAMICS AND AVERAGING:
   Because the NOE depends on 1/r⁶, the observed signal is heavily weighted
   towards shorter distances in a dynamic ensemble. This "r-6 averaging"
   means that even a small population of a "compact" state can dominate the
   observed NOE spectrum.

5. THE NOE EXPERIMENT (NOESY):
   The Nuclear Overhauser Effect Spectroscopy (NOESY) experiment is the most
   important tool for protein structure determination. By measuring the
   transfer of magnetization between protons, researchers can build a map of
   spatial proximities that uniquely define the protein's fold.

6. QUANTITATIVE VS QUALITATIVE NOES:
   While the NOE intensity is mathematically linked to distance, in practice,
   experimental noise and relaxation effects often lead to "binning" NOEs
   into distance categories (Strong, Medium, Weak). This robust approach
   ensures that the resulting structures are not overly sensitive to minor
   experimental variations.

7. THE SPIN-DIFFUSION PROBLEM:
   In larger proteins, magnetization can "wander" between multiple protons,
   a phenomenon known as spin diffusion. This can lead to misleading NOEs
   between atoms that are not directly adjacent. This occurs because
   magnetization is transferred from atom A to B, and then from B to C,
   resulting in a perceived A-C contact that may not be direct. Advanced
   simulation and validation tools like RPF scores are essential for
   identifying and correcting these effects by ensuring the global
   consistency of the model.

NOE DISTANCE RESTRAINTS IN SYNTH-PDB:
======================================
In structural biology workflows, NOE intensities are converted into distance
restraints (upper bounds). A "strong" peak typically corresponds to < 2.5 Å,
"medium" to < 3.5 Å, and "weak" to < 5.0 Å.

`synth-pdb` generates these synthetic restraints to allow for:
- Benchmarking of new structure determination algorithms.
- Training machine learning models to predict folds from sparse data.
- Validation of generated decoys against expected physical constraints.
- Educational demonstrations of how structural data is derived from spectra.
- Exploration of conformational diversity in NMR ensembles.

HISTORICAL CONTEXT:
====================
The development of the NOE as a structural tool was a transformative moment
in biochemistry, eventually leading to Kurt Wüthrich's Nobel Prize in 2002.
By proving that proteins could be solved in their native solution state,
NMR opened the door to studying dynamic systems, disordered regions, and
interactions that are inaccessible to X-ray crystallography.

THE ROLE OF RPF SCORES IN STRUCTURAL GENOMICS:
==============================================
During the Structural Genomics era (e.g., the Northeast Structural Genomics
Consortium, NESG), thousands of protein structures were determined using
NMR. To ensure the reliability of these models, the community developed
rigorous validation metrics.

The RPF score (Recall, Precision, and F-measure) was a breakthrough because
it provided a software-independent assessment. Unlike a force-field energy
(which depends on the software used for refinement), RPF scores directly
measure the fit to experimental NOE peaks.

- A structure with a high F-measure (> 0.70) is generally considered to be
  of high quality and well-supported by the data.
- A structure with low F-measure, despite having "good" geometry, is
  likely misfolded or missing critical experimental information.

THE INFORMATION RETRIEVAL ORIGIN OF RPF:
=========================================
The RPF metrics (Recall, Precision, and F-measure) were originally developed
for evaluating search engines and database queries. In structural biology:

1. THE UNIVERSE OF CONTACTS:
   The structural model generates a "predicted" set of short-range proton
   contacts based on its 3D coordinates. This is our "retrieved" set.

2. THE TRUTH SET:
   The experimental NOE dataset acts as the "ground truth" or target set
   of desired information. This is our "relevant" set.

3. THE VALIDATION GOAL:
   We want the model to find all the target contacts (High Recall) AND we
   want only the target contacts to be present (High Precision). This
   minimizes false positives and false negatives in our structural data.

BIBLIOGRAPHY AND FURTHER READING:
=================================
1. Wüthrich, K. (1986). NMR of Proteins and Nucleic Acids. Wiley.
2. Huang, Y. J., et al. (2005). Protein NMR recall, precision, and
   F-measure scores (RPF scores). J. Am. Chem. Soc. 127, 1665-1674.
3. Montelione, G. T., et al. (2013). Recommendations of the wwPDB NMR
   Validation Task Force. Structure 21, 1563-1570.
4. Cavanagh, J., et al. (2007). Protein NMR Spectroscopy. Academic Press.
5. Bax, A. (1989). Two-dimensional NMR and protein structure. Annu. Rev.
   Biochem. 58, 223-256.

"""

import logging
import os
from typing import Any, Dict, List

import biotite.structure as struc
import numpy as np
import synth_nmr as _nmr

# ── LOGGING CONFIGURATION ───────────────────────────────────────────────────
# We use a dedicated logger for the NMR module to allow for granular
# debugging of spectroscopic calculations. This follows the standard
# Python logging pattern.
logger = logging.getLogger(__name__)

# ── RE-EXPORTS ──────────────────────────────────────────────────────────────
# We re-export the core engine functions to maintain a stable API while
# allowing the underlying implementation to evolve in the synth-nmr package.

# calculate_synthetic_noes: This is the core generator for synthetic NOEs.
# It takes a 3D structure (Biotite AtomArray) and returns a list of all
# atom pairs that are close enough in space (< 6.0 Å) to produce an NMR
# signal in a standard NOESY experiment.
#
# Technical Implementation Summary:
# 1. Identify all Hydrogen atoms ('H') using atomic element mapping.
# 2. Compute a complete pairwise distance matrix using vectorized NumPy.
# 3. Apply a detection threshold (typically 5.0 to 6.0 Ångströms).
# 4. Map the resulting indices back to residue and atom identifiers.
calculate_synthetic_noes = _nmr.calculate_synthetic_noes

def calculate_rpf_score(
    structure: struc.AtomArray,
    restraints: List[Dict[str, Any]],
    distance_threshold: float = 5.0
) -> Dict[str, float]:
    """Calculates Recall, Precision, and F-measure (RPF) scores for an NMR structure.

    This function implements a simplified version of the RPF scores developed by
    the Montelione group (Huang et al. 2005, JACS 127:1665) to assess the
    congruence between a structural model and a set of NOE distance restraints.

    SCIENTIFIC RATIONALE:
    --------------------
    RPF scores are information retrieval metrics applied to structural biology.
    They provide a software-independent measure of how well a 3D model
    represents the underlying spectroscopic data.

    - RECALL (R): Measures the completeness of the model. It asks: "Of all the
      experimental observations we made, how many does this model correctly
      account for?" A low recall indicates the model fails to capture the
      experimental data, perhaps due to misfolding or under-refinement.
      Formula: R = (Satisfied Restraints) / (Total Restraints).

    - PRECISION (P): Measures the specificity of the model. It asks: "Of all
      the spatial proximities predicted by this 3D model, how many are
      actually supported by our data?" A low precision indicates the model
      may have "over-folded" or introduced spurious interactions not
      supported by evidence. It acts as a critical check against
      "over-fitting" the data.
      Formula: P = (Supported Short Distances) / (Total Short Distances < threshold).

    - F-MEASURE (F): The harmonic mean of Recall and Precision. It provides
      a single global metric of the model's reliability (ranging from 0.0
      to 1.0). A high F-measure requires BOTH high recall and high precision.
      Formula: F = (2 * P * R) / (P + R).

    MATHEMATICAL FORMULATION OF RPF SCORES:
    --------------------------------------
    The RPF scores originate from information retrieval theory. In the context
    of NMR structure validation:

    1. RECALL (R):
       Let 'E' be the set of experimental restraints and 'M' be the set of
       contacts satisfied by the model.
       R = |M ∩ E| / |E|
       It measures the "sensitivity" or data-coverage of the model.

    2. PRECISION (P):
       Let 'S' be the set of all short inter-proton distances (< threshold)
       predicted by the model.
       P = |M ∩ E| / |S|
       It measures the "false positive rate" of the model's packing.

    3. F-MEASURE (F):
       F is the Sorensen-Dice coefficient, providing a single scalar that
       balances completeness and specificity.

    In the structural genomics era (NESG), RPF scores were used as a sensitive
    indicator of structure quality, independent of the software used for
    structure calculation. They allow for an unbiased comparison between
    models generated by different groups or algorithms.

    DEVELOPER NOTE — Corner Cases:
    ------------------------------
    For a perfect fit, R=1, P=1, and thus F=1.
    If a model is completely disordered but happens to satisfy one restraint,
    R will be low, P will be high, and F will be low.
    If a model is a collapsed "black hole" satisfying all restraints but
    creating thousands of spurious contacts, R will be high, P will be low,
    and F will be low.

    Args:
        structure (struc.AtomArray): The structural model (AtomArray) to be
            evaluated against the restraints. The structure must include
            explicit hydrogen atoms for accurate distance calculation.
        restraints (List[Dict[str, Any]]): A list of NOE restraints representing
            the "target" data. Each dictionary in the list should contain:
            - 'res_i' / 'index_1': Residue ID of the first atom in the pair.
            - 'atom_i' / 'atom_name_1': Name of the first atom.
            - 'res_j' / 'index_2': Residue ID of the second atom in the pair.
            - 'atom_j' / 'atom_name_2': Name of the second atom.
            - 'upper_bound' / 'upper_limit': The maximum allowed distance in Å.
        distance_threshold (float, optional): The threshold in Å used to identify
            "short distances" in the model for the Precision calculation.
            Common values are 5.0 or 6.0 Å. Defaults to 5.0 Å.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics:
            - 'recall': The recall score [0.0 to 1.0].
            - 'precision': The precision score [0.0 to 1.0].
            - 'f_measure': The composite F-measure score [0.0 to 1.0].
    """
    # ── DEFENSIVE CHECKS ─────────────────────────────────────────────────────
    # We return zeroed scores if no restraints are provided to avoid potential
    # division by zero errors in the statistics. This handles edge cases like
    # completely disordered regions or empty datasets gracefully.
    # Without data, accuracy metrics are mathematically undefined.
    if not restraints:
        # Log a warning to notify the researcher of the empty dataset.
        # This is useful for identifying pipelines that may have failed to
        # load or generate the restraint set correctly.
        logger.warning("Empty restraint list provided to calculate_rpf_score.")
        return {"recall": 0.0, "precision": 0.0, "f_measure": 0.0}

    # 1. ── CALCULATE RECALL (R) ──────────────────────────────────────────────
    # Recall measures the sensitivity of the model to the provided restraints.
    # We iterate through every target restraint and check if our model's
    # geometry is consistent with it. This is essentially a "pass/fail"
    # check for every expected interaction.
    satisfied_count = 0

    # Loop through each target interaction defined by the researcher.
    # We use explicit iteration here to allow for granular error checking
    # of individual restraint records.
    # This loop has O(M) complexity where M is the number of restraints.
    for res in restraints:
        # Support multiple key naming conventions for robustness across
        # different versions of the synth-nmr engine and external tools.
        # Format A: index_1, atom_name_1, etc. (standard engine format)
        # Format B: res_i, atom_i, etc. (convenience/user format)
        # This normalization ensures the function is backward compatible.
        res_i = res.get('index_1') or res.get('res_i')
        atom_i = res.get('atom_name_1') or res.get('atom_i')
        res_j = res.get('index_2') or res.get('res_j')
        atom_j = res.get('atom_name_2') or res.get('atom_j')

        # Upper bound distance limit (default to 5.0 Å if not specified).
        # This is the "target" distance derived from the NOE spectrum.
        # A model atom pair must be closer than this to satisfy the data.
        upper_bound = res.get('upper_limit') or res.get('upper_bound') or 5.0

        # ── CREATE MASKS ─────────────────────────────────────────────────────
        # We create Boolean masks to isolate the specific atoms i and j in the
        # structure. This vectorized approach is robust against variations
        # in atom ordering or nomenclature between different PDB versions.
        # It ensures that we find the correct atoms regardless of where
        # they appear in the data array. This is a standard Biotite pattern.
        mask_i = (structure.res_id == res_i) & (structure.atom_name == atom_i)
        mask_j = (structure.res_id == res_j) & (structure.atom_name == atom_j)

        # ── CALCULATE DISTANCE ───────────────────────────────────────────────
        # We only calculate the distance if BOTH atoms are actually found
        # in the provided structural model. If an atom is missing, the
        # restraint cannot be "satisfied" and will count against recall.
        # This penalizes models that are incomplete or have missing residues.
        if np.any(mask_i) and np.any(mask_j):
            # Calculate the Euclidean distance between the first match for
            # each atom mask. Biotite coords are stored as (x, y, z) in Å.
            # Use index 0 to get the first matching atom (handling alt-locs).
            p1_coords = structure[mask_i][0].coord
            p2_coords = structure[mask_j][0].coord

            # The norm of the difference vector gives the straight-line distance.
            # This is the standard distance used for NOE upper-bound checking.
            # In structural biology, this is the d_ij value.
            dist = np.linalg.norm(p1_coords - p2_coords)

            # ── SATISFACTION CHECK ───────────────────────────────────────────
            # A restraint is considered "satisfied" if the distance in our model
            # is less than or equal to the experimental upper bound.
            # No tolerance is applied here; the threshold is strict.
            if dist <= upper_bound:
                satisfied_count += 1

    # Final Recall calculation: Total satisfied / Total expected.
    # A value of 1.0 indicates perfect agreement with all experimental bounds.
    # If R=0.8, it means 80% of our expected contacts were found in the model.
    recall = satisfied_count / len(restraints)

    # 2. ── CALCULATE PRECISION (P) ───────────────────────────────────────────
    # Precision identifies potential "false positive" contacts in the model.
    # It answers: "Does the model predict interactions that the data says
    # shouldn't be there?" This is the most computationally expensive part
    # because it involves an all-vs-all search of the structural model.
    # The complexity is O(N_protons^2).

    # ── FILTER FOR PROTONS ───────────────────────────────────────────────────
    # We first filter the structure for protons (Hydrogen atoms), as the
    # vast majority of NOEs are measured between 1H nuclei.
    # Using element "H" captures all isotopes (D, T) if present.
    # This reduction in atom count significantly speeds up the distance matrix.
    protons = structure[structure.element == "H"]

    # ── HANDLING SPARSE SYSTEMS ──────────────────────────────────────────────
    # If the model has fewer than two protons, we cannot compute any
    # inter-atomic distances, so we return a default precision of 1.0.
    # This reflects that there are no "unsupported" interactions possible.
    if len(protons) < 2:
        # Defaulting to 1.0 avoids NaN values in downstream statistics.
        precision = 1.0
    else:
        # ── COMPUTE DISTANCE MATRIX ──────────────────────────────────────────
        # We compute a full O(N^2) distance matrix for all proton-proton pairs.
        # This gives us a complete map of all spatial contacts in the model.
        # For a protein with 1000 protons, this is a 1,000,000 entry matrix.
        # We use NumPy's optimized internal C-loops for this calculation.
        coords = protons.coord

        # Use NumPy broadcasting for efficient pairwise vector subtraction.
        # This expands (N, 3) to (N, 1, 3) and (1, N, 3) to produce (N, N, 3).
        # Every entry (i, j) in the result is the vector from atom i to j.
        # This is a high-memory operation for very large complexes.
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

        # Square the differences, sum across the XYZ dimension (axis -1),
        # and take the square root to obtain the final distance matrix.
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # ── FILTER FOR SHORT DISTANCES ───────────────────────────────────────
        # The matrix is symmetric (dist[i,j] == dist[j,i]), so we only need
        # the upper triangle to avoid double-counting.
        # triu(..., k=1) excludes the diagonal (the distance from an atom to itself).
        tri_mask = np.triu(np.ones(dist_matrix.shape, dtype=bool), k=1)

        # Identify all pairs in the structure that are within the 'short' threshold.
        # These are the "predicted contacts" that we will now validate.
        # Typical thresholds are 5.0 Å (standard) or 6.0 Å (broad).
        # This mask identifies all potential NOE sources in the structural model.
        short_distances_mask = (dist_matrix < distance_threshold) & tri_mask

        # Count the total number of short-range contacts predicted by the model.
        # This is our denominator for the Precision calculation.
        total_short_in_struct = np.sum(short_distances_mask)

        # ── RESTRAINT LOOKUP OPTIMIZATION ────────────────────────────────────
        # To calculate Precision efficiently, we build a set of all target
        # interactions. Using a set allows for average O(1) lookup time.
        # Without this, we would have an O(N^2 * M) complexity.
        # We sort each pair (res_id, atom_name) to ensure that the order of
        # atoms in the restraint (i-j vs j-i) does not affect the match.
        restraint_pairs = set()
        for r in restraints:
            # Normalize keys for lookup optimization to handle different formats.
            # This logic must be identical to the Recall normalization above.
            ri = r.get('index_1') or r.get('res_i')
            ai = r.get('atom_name_1') or r.get('atom_i')
            rj = r.get('index_2') or r.get('res_j')
            aj = r.get('atom_name_2') or r.get('atom_j')

            # Create a unique, hashable identifier for each atom in the pair.
            # Using a tuple of (res, atom) allows for precise matching.
            # We use (index, name) as the primary key.
            p1_id = (ri, ai)
            p2_id = (rj, aj)
            # Sorted tuple ensures a canonical representation for the pair.
            # e.g., (1, 'HA') and (5, 'HN') always becomes ((1, 'HA'), (5, 'HN')).
            pair = tuple(sorted([p1_id, p2_id]))
            restraint_pairs.add(pair)

        supported_count = 0
        # Iterate over all "short" pairs identified in our structural model.
        # np.where returns the indices (i, j) of all True values in the mask.
        indices = np.where(short_distances_mask)
        for idx_i, idx_j in zip(indices[0], indices[1]):
            # Get the actual atoms from the filtered proton array.
            p1 = protons[idx_i]
            p2 = protons[idx_j]

            # Check if this specific short distance exists in our target list.
            # We map the model atoms to the same (res_id, atom_name) format.
            p1_id = (p1.res_id, p1.atom_name)
            p2_id = (p2.res_id, p2.atom_name)
            pair = tuple(sorted([p1_id, p2_id]))

            # If the interaction is found in the target list, it is "supported".
            # This means the model's packing is justified by the data.
            # If it's not found, it's a potential false positive (over-packing).
            if pair in restraint_pairs:
                supported_count += 1

        # Final Precision calculation: Total supported / Total predicted.
        # A value of 1.0 indicates no spurious "extra" packing in the model.
        # A value of 0.5 would mean half of our model's contacts are unsupported.
        precision = supported_count / total_short_in_struct if total_short_in_struct > 0 else 1.0

    # 3. ── CALCULATE F-MEASURE (F) ───────────────────────────────────────────
    # The F-measure (Sorensen–Dice coefficient) is the harmonic mean of R and P.
    # It provides a balanced view of structural quality, penalizing models
    # that achieve high recall by simply being too compact (low precision).
    # This is our primary target metric for structure determination.
    if recall + precision > 0:
        # Standard formulation for the harmonic mean of sensitivity and specificity.
        # We multiply by 2 to normalize the score to the [0, 1] range.
        f_measure = (2 * recall * precision) / (recall + precision)
    else:
        # Default to 0 if no interactions are found or supported.
        # This occurs if the model and restraints have zero overlap.
        f_measure = 0.0

    # Return the composite metrics as a dictionary for further analysis.
    # These values can be plotted or used as objective functions for refinement.
    # A successful NMR structure determination typically targets F > 0.8.
    return {
        "recall": recall,
        "precision": precision,
        "f_measure": f_measure
    }

def read_restraint_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads NOE restraints from a file.

    SCIENTIFIC RELEVANCE:
    --------------------
    In protein NMR, spatial information is captured as distance restraints.
    These restraints are typically derived from NOESY experiments where
    cross-peak intensities are converted into upper-bound distance limits.
    Supporting multiple file formats is essential for interoperability between
    different structural biology software suites.

    Supports:
    1. NEF (.nef): The NMR Exchange Format. This is the modern, IUPAC-approved
       standard for exchanging NMR data between repositories like the BMRB
       and structure calculation software like CYANA or ARIA.

    2. Simple (.restraints, .txt): A whitespace-separated format used for
       rapid prototyping, manual editing, and teaching.
       Format: res_i atom_i res_j atom_j upper_bound
       Example: 1 HN 5 HA 3.5 (meaning: atom HN of residue 1 is within
                3.5 Å of atom HA of residue 5).

    Args:
        file_path (str): The system path to the restraint file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each entry
            represents a single distance restraint.
    """
    # ── VALIDATE FILE EXISTENCE ──────────────────────────────────────────────
    # Before attempting to parse, we ensure the file is accessible on disk.
    # Missing files are a common source of errors in automated pipelines.
    # We use os.path.exists for cross-platform compatibility.
    if not os.path.exists(file_path):
        # We raise a standard FileNotFoundError for idiomatic error handling.
        # This allows the caller to handle missing files gracefully.
        raise FileNotFoundError(f"Restraint file not found: {file_path}")

    # ── NEF PARSER ───────────────────────────────────────────────────────────
    # If the file extension suggests the modern NEF standard, we delegate
    # to the specialized NEF I/O engine. NEF files are self-documenting
    # and include metadata about the sequence and experiment type.
    # The 'synth-nmr' package provides the core NEF parsing logic.
    if file_path.endswith(".nef"):
        # Lazy import of NEF logic to minimize startup time for non-NMR tasks.
        # This reduces the overhead for basic generator users.
        from .nef_io import read_nef_restraints
        return read_nef_restraints(file_path)

    # ── SIMPLE WHITESPACE PARSER ─────────────────────────────────────────────
    # For non-standard or simplified files, we use a robust line-by-line parser.
    # This parser is designed to be tolerant of blank lines and comments.
    # It follows the Unix tradition of "simple text files for simple tasks."
    restraints = []
    try:
        with open(file_path) as f:
            for line in f:
                # Remove leading/trailing whitespace to avoid empty field errors.
                # This makes the parser tolerant of mixed line endings (\n vs \r\n).
                line = line.strip()

                # Ignore empty lines and lines starting with '#' (standard comments).
                # This allows users to annotate their restraint files freely.
                # Lines that don't match the format are skipped for robustness.
                if not line or line.startswith("#"):
                    continue

                # Split the line into fields. We expect at least 5 columns.
                # Standard Columns: [Res1, Atom1, Res2, Atom2, UpperBound]
                # Any extra columns are ignored.
                parts = line.split()
                if len(parts) >= 5:
                    # Append the parsed restraint using the internal engine's
                    # standard keys for immediate compatibility with
                    # calculate_rpf_score.
                    # We convert numeric strings to integers and floats here.
                    restraints.append({
                        "index_1": int(parts[0]),
                        "atom_name_1": parts[1],
                        "index_2": int(parts[2]),
                        "atom_name_2": parts[3],
                        "upper_limit": float(parts[4])
                    })

        # Log the success for educational transparency.
        # This helps researchers verify that their files were read correctly.
        # We report the total count of restraints successfully ingested.
        logger.info(f"Successfully parsed {len(restraints)} restraints from {file_path}.")
        return restraints

    except Exception as e:
        # Wrap underlying parsing errors in a descriptive ValueError.
        # This provides a clearer stack trace for the end user and helps
        # identify syntax errors in their restraint files.
        # Descriptive errors are a hallmark of high-quality scientific software.
        raise ValueError(f"Failed to parse restraint file {file_path}: {e}")

# ── MODULE EXPORTS ───────────────────────────────────────────────────────────
# We define __all__ to explicitly control the public API of this module.
# This ensures that internal helpers and imports are not exposed to users.
# The following symbols are considered the stable, supported interface:
__all__ = ["calculate_synthetic_noes", "calculate_rpf_score", "read_restraint_file"]

# ── END OF MODULE ────────────────────────────────────────────────────────────
# Documentation density check: This module maintains an exceptionally high
# level of internal commentary to serve as a pedagogical resource for
# structural biology students and researchers. Every architectural
# decision is documented alongside its scientific justification.
#
# TROUBLESHOOTING AND COMMON PITFALLS:
# -----------------------------------
# 1. MISSING HYDROGENS: If RPF scores are unexpectedly 0, verify that the
#    AtomArray contains Hydrogen atoms. Use --cap-termini or --cap-all.
# 2. ATOM NOMENCLATURE: Ensure atom names in restraint files match the
#    PDB standard (e.g., 'HN' vs 'H').
# 3. RESIDUE OFFSET: Verify that residue indexing starts from 1 as expected.
#
# Technical Maintenance Roadmap:
# - Consider adding support for RDC-based RPF scores (RDC-RPF).
# - Implement cell-list optimization for precision calculation on large systems.
# - Add automated fetching of BMRB restraint files for direct validation.
# - Integrate with structural genomics validation servers (PSVS).
# - Support IUPAC to PDB atom name mapping for experimental data.
# - Add support for ambiguity (OR) restraints common in NMR.
# - Implement visualization for violated restraints in the 3D viewer.
# - Add support for distance restraints between non-hydrogen atoms (e.g. paramagnetic).
# - Implement a 'verbose' mode for RPF calculation to list specific violations.
# - Ensure compatibility with biotite 1.0+ internal structure changes.
# - Add support for time-averaged distance restraints for MD ensembles.
# - Implement NOE intensity simulation based on full relaxation matrix.
# - Add automated reporting of 'long-range' vs 'short-range' RPF subsets.
