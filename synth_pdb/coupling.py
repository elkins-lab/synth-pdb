"""Coupling utilities for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_hn_ha_coupling, predict_couplings_from_structure

The J-coupling (scalar coupling) is a fundamental NMR parameter that provides
information about the chemical bonding and local geometry of a molecule.
In proteins, the 3J(HN-HA) coupling is particularly useful as it relates to the
phi torsion angle via the Karplus equation.

See: https://github.com/elkins/synth-nmr
"""

import typing

import biotite.structure as struc
import numpy as np
import synth_nmr.j_coupling as _j
from synth_nmr.structure_utils import get_residue_info

calculate_hn_ha_coupling = _j.calculate_hn_ha_coupling_from_phi


def predict_couplings_from_structure(
    structure: typing.Any,
) -> dict[str, dict[int, float]]:
    """Predict 3J_HNHa coupling constants from a Biotite AtomArray.

    This function wraps the synth-nmr engine and provides biophysical enhancements:
    1. Filters out residues that physically lack a backbone amide proton (e.g., Proline).
    2. Corrects the Karplus equation phase for D-amino acids by inverting the phi angle.

    EDUCATIONAL NOTE - The Karplus Equation and 3J(HN-HA) Couplings
    ================================================================
    The 3J(HN-HA) scalar coupling is a through-bond magnetic interaction between
    the backbone amide proton (HN) and the alpha proton (HA). This coupling is
    highly sensitive to the intervening H-N-CA-H dihedral angle (theta), which is
    geometrically related to the backbone Ramachandran Phi (phi) angle.

    The relationship is empirically described by the Karplus equation:
        3J(theta) = A * cos^2(theta) - B * cos(theta) + C

    For typical L-amino acids, theta ~ phi - 60deg. This means:
    - In an ideal alpha-helix (phi ~ -60deg), theta ~ -120deg, yielding small couplings (3-5 Hz).
    - In a beta-sheet (phi ~ -120deg to -140deg), theta ~ 180deg, yielding large couplings (8-10 Hz).

    EDUCATIONAL NOTE - Proline and Secondary Amines
    ===============================================
    Proline (PRO) and its D-isomer (DPR) are unique among standard amino acids
    because their sidechain cyclizes onto the backbone nitrogen, forming a
    secondary amine. Consequently, the nitrogen atom lacks an attached hydrogen
    when incorporated into a peptide bond. Without an amide proton, there is no
    H-N-CA-H spin system, and thus the 3J(HN-HA) coupling is physically undefined.
    Our shim layer explicitly strips these residues from the results.

    EDUCATIONAL NOTE - D-Amino Acids and Stereochemistry
    ====================================================
    D-amino acids are non-superimposable mirror images of natural L-amino acids.
    Because the Karplus equation is parameterized for the natural L-configuration,
    directly feeding a D-amino acid's phi angle into the standard Karplus curve
    yields incorrect results (e.g., predicting ~6.3 Hz instead of ~3.8 Hz for a
    D-alpha helix where phi ~ +60deg).

    Due to mirror symmetry, the geometric relationship inverts:
    theta_D(phi) = -theta_L(-phi). Therefore, we can accurately predict the D-amino acid
    coupling by evaluating the L-parameterized Karplus equation at the
    negated phi angle: J_D(phi) = J_L(-phi).

    Args:
        structure: Biotite AtomArray

    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz).
    """
    # -- 1. PRIMARY PREDICTION ------------------------------------------------
    # We first delegate to the core synth-nmr engine to calculate raw couplings.
    # The engine uses a highly optimized C++ backend or Numba-jitted array ops
    # to evaluate the Karplus equation for all residues simultaneously.
    raw_couplings = _j.calculate_hn_ha_coupling(structure)
    filtered_couplings: dict[str, dict[int, float]] = {}

    # -- 2. BIOPHYSICAL FILTER DEFINITIONS ------------------------------------
    # Proline-like residues do not have an amide proton when in a peptide bond.
    # Therefore, they cannot produce a 3J_HNHa coupling.
    proline_names = {"PRO", "DPR"}

    # We maintain an exhaustive list of D-amino acids (the D-stereoisomers).
    # This allows us to intercept them and perform the requisite stereochemical
    # inversion on their backbone phi angles before evaluating Karplus.
    d_amino_acids = {
        "DAL",
        "DAR",
        "DAN",
        "DAS",
        "DCY",
        "DGL",
        "DGN",
        "DHI",
        "DIL",
        "DLE",
        "DLY",
        "DME",
        "DPH",
        "DPR",
        "DSE",
        "DTH",
        "DTR",
        "DTY",
        "DVA",
    }

    # -- 3. PRE-CALCULATING STRUCTURAL ANGLES ---------------------------------
    # We extract the entire array of backbone phi dihedrals from the structure.
    # Biotite returns these in radians, so we convert them to degrees, which
    # the Karplus parametrization expects.
    phi, _, _ = struc.dihedral_backbone(structure)
    chain_ids, res_ids, _, _ = get_residue_info(structure)

    # Build a rapid-lookup dictionary for phi angles keyed by (chain_id, res_id).
    # We explicitly cast res_id to int to ensure type-matching with raw_couplings.
    phi_map = {}
    for c, r, p in zip(chain_ids, res_ids, phi, strict=False):
        phi_map[(c, int(r))] = np.degrees(p)

    # -- 4. ITERATIVE FILTERING & CORRECTION ----------------------------------
    # We iterate over the raw predictions and selectively copy/modify them
    # into our sanitized output dictionary.
    for chain_id, res_dict in raw_couplings.items():
        filtered_chain = {}
        for res_id, j_val in res_dict.items():
            # Check residue name from structure by generating a boolean mask.
            # We select the first matching atom's residue name.
            res_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
            if not res_mask.any():
                continue
            res_name = structure.res_name[res_mask][0]

            # Physics check: Proline has no HN proton; skip entirely.
            if res_name in proline_names:
                continue

            # Stereochemistry check: D-amino acids require phase inversion.
            if res_name in d_amino_acids:
                p_angle = phi_map.get((chain_id, res_id))
                if p_angle is not None and not np.isnan(p_angle):
                    # J_D(phi) = J_L(-phi) due to stereochemical inversion.
                    # We round to 2 decimal places to match typical NMR precision.
                    j_val = float(round(_j.calculate_hn_ha_coupling_from_phi(-p_angle), 2))

            filtered_chain[res_id] = j_val

        # Only append chains that contain valid couplings.
        if filtered_chain:
            filtered_couplings[chain_id] = filtered_chain

    return filtered_couplings


__all__ = [
    "calculate_hn_ha_coupling",
    "predict_couplings_from_structure",
]
