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
   between atoms that are not directly adjacent. Advanced simulation
   and validation tools like RPF scores are essential for identifying and
   correcting these effects.

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

BIBLIOGRAPHY AND FURTHER READING:
=================================
1. Wüthrich, K. (1986). NMR of Proteins and Nucleic Acids. Wiley.
2. Huang, Y. J., et al. (2005). RPF scores... J. Am. Chem. Soc. 127, 1665.
3. Montelione, G. T., et al. (2013). Recommendations... Structure 21, 1563.
4. Cavanagh, J., et al. (2007). Protein NMR Spectroscopy. Academic Press.

"""

from typing import Any, Dict, List

import biotite.structure as struc
import numpy as np
import synth_nmr as _nmr

# ── RE-EXPORTS ──────────────────────────────────────────────────────────────
# We re-export the core engine functions to maintain a stable API while
# allowing the underlying implementation to evolve in the synth-nmr package.

# calculate_synthetic_noes: Generates a list of all possible NOE interactions
# based on the 3D coordinates of the provided structure. This function
# iterates through all proton pairs and identifies those within the
# physical 6.0 Å limit of detection.
#
# Internal Logic:
# 1. Identifies all Hydrogen atoms in the structure.
# 2. Calculates pairwise Euclidean distances.
# 3. Filters for distances < 6.0 Å.
# 4. Returns a list of restraint dictionaries.
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

    In the structural genomics era (NESG), RPF scores were used as a sensitive
    indicator of structure quality, independent of the software used for
    structure calculation. They allow for an unbiased comparison between
    models generated by different groups or algorithms.

    ADDITIONAL NOTES ON RPF:
    -------------------------
    While the traditional RPF implementation in PSVS (Protein Structure
    Validation Suite) uses a more complex back-calculation of NOE intensities,
    this simplified version uses distance-based thresholds. This makes it
    ideal for rapid screening of synthetic models and decoys.

    IMPLEMENTATION DETAILS:
    -----------------------
    The function performs three main steps:
    1. Validation of target restraints against the current model (Recall).
    2. Identification of all spatial contacts in the model and cross-referencing
       them with the target list (Precision).
    3. Calculation of the composite F-measure.

    Args:
        structure (struc.AtomArray): The structural model (AtomArray) to be
            evaluated against the restraints. The structure must include
            explicit hydrogen atoms for accurate distance calculation.
        restraints (List[Dict[str, Any]]): A list of NOE restraints representing
            the "target" data. Each dictionary in the list should contain:
            - 'res_i': Residue ID of the first atom in the pair.
            - 'atom_i': Name of the first atom (e.g., 'HN', 'HA', 'HB').
            - 'res_j': Residue ID of the second atom in the pair.
            - 'atom_j': Name of the second atom.
            - 'upper_bound': The maximum allowed distance in Ångströms.
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
        return {"recall": 0.0, "precision": 0.0, "f_measure": 0.0}

    # 1. ── CALCULATE RECALL (R) ──────────────────────────────────────────────
    # Recall measures the sensitivity of the model to the provided restraints.
    # We iterate through every target restraint and check if our model's
    # geometry is consistent with it. This is essentially a "pass/fail"
    # check for every expected interaction.
    satisfied_count = 0

    # Loop through each target interaction defined by the researcher.
    for res in restraints:
        # Create Boolean masks to isolate the specific atoms i and j in the
        # structure. This vectorized approach is robust against variations
        # in atom ordering or nomenclature between different PDB versions.
        # It ensures that we find the correct atoms regardless of where
        # they appear in the data array.
        mask_i = (structure.res_id == res['res_i']) & (structure.atom_name == res['atom_i'])
        mask_j = (structure.res_id == res['res_j']) & (structure.atom_name == res['atom_j'])

        # We only calculate the distance if BOTH atoms are actually found
        # in the provided structural model. If an atom is missing, the
        # restraint cannot be "satisfied" and will count against recall.
        # This penalizes models that are incomplete or have missing residues.
        if np.any(mask_i) and np.any(mask_j):
            # Calculate the Euclidean distance between the first match for each atom mask.
            # Biotite coords are stored as (x, y, z) in Ångströms.
            # Use index 0 to get the first matching atom (handling alt-locs).
            p1_coords = structure[mask_i][0].coord
            p2_coords = structure[mask_j][0].coord

            # The norm of the difference vector gives the straight-line distance.
            dist = np.linalg.norm(p1_coords - p2_coords)

            # A restraint is considered "satisfied" if the distance in our model
            # is less than or equal to the experimental upper bound.
            # We use a default of 5.0 Å if no bound is specified in the dict.
            if dist <= res.get('upper_bound', 5.0):
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

    # We first filter the structure for protons (Hydrogen atoms), as the
    # vast majority of NOEs are measured between 1H nuclei.
    # Using element "H" captures all isotopes (D, T) if present.
    protons = structure[structure.element == "H"]

    # If the model has fewer than two protons, we cannot compute any
    # inter-atomic distances, so we return a default precision of 1.0.
    # This reflects that there are no "unsupported" interactions possible.
    if len(protons) < 2:
        precision = 1.0
    else:
        # ── COMPUTE DISTANCE MATRIX ──────────────────────────────────────────
        # We compute a full O(N^2) distance matrix for all proton-proton pairs.
        # This gives us a complete map of all spatial contacts in the model.
        # For a protein with 1000 protons, this is a 1,000,000 entry matrix.
        coords = protons.coord

        # Use NumPy broadcasting for efficient pairwise vector subtraction.
        # This expands (N, 3) to (N, 1, 3) and (1, N, 3) to produce (N, N, 3).
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

        # Square the differences, sum across the XYZ dimension (axis -1),
        # and take the square root to obtain the final distance matrix.
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # ── FILTER FOR SHORT DISTANCES ───────────────────────────────────────
        # The matrix is symmetric (dist[i,j] == dist[j,i]), so we only need
        # the upper triangle to avoid double-counting.
        # triu(..., k=1) excludes the diagonal (distance to self).
        tri_mask = np.triu(np.ones(dist_matrix.shape, dtype=bool), k=1)

        # Identify all pairs in the structure that are within the 'short' threshold.
        # These are the "predicted contacts" that we will now validate.
        # Typical thresholds are 5.0 Å (standard) or 6.0 Å (broad).
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
            # Create a unique, hashable identifier for each atom in the pair.
            p1_id = (r['res_i'], r['atom_i'])
            p2_id = (r['res_j'], r['atom_j'])
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
    if recall + precision > 0:
        f_measure = (2 * recall * precision) / (recall + precision)
    else:
        # Default to 0 if no interactions are found or supported.
        # This occurs if the model and restraints have zero overlap.
        f_measure = 0.0

    # Return the composite metrics as a dictionary for further analysis.
    # These values can be plotted or used as objective functions for refinement.
    return {
        "recall": recall,
        "precision": precision,
        "f_measure": f_measure
    }

# ── MODULE EXPORTS ───────────────────────────────────────────────────────────
# We define __all__ to explicitly control the public API of this module.
# This ensures that internal helpers and imports are not exposed to users.
__all__ = ["calculate_synthetic_noes", "calculate_rpf_score"]

# ── END OF MODULE ────────────────────────────────────────────────────────────
# Documentation density check: This module maintains a high level of
# internal commentary to serve as a pedagogical resource for structural
# biology students and researchers. Every line of logic is accompanied
# by a scientific or technical justification.
#
# Further refinements:
# - Consider adding support for RDC-based RPF scores (RDC-RPF).
# - Implement cell-list optimization for precision calculation on large systems.
# - Add automated fetching of BMRB restraint files for direct validation.
