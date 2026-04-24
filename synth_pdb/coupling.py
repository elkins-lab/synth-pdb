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
) -> typing.Dict[str, typing.Dict[int, float]]:
    """Predict 3J_HNHa coupling constants from a Biotite AtomArray.

    This function wraps the synth-nmr engine and provides biophysical enhancements:
    1. Filters out residues that physically lack a backbone amide proton (e.g., Proline).
    2. Corrects the Karplus equation phase for D-amino acids by inverting the phi angle.

    EDUCATIONAL NOTE — The Karplus Equation and 3J(HN-HA) Couplings
    ================================================================
    The 3J(HN-HA) scalar coupling is a through-bond magnetic interaction between
    the backbone amide proton (HN) and the alpha proton (HA). This coupling is
    highly sensitive to the intervening H-N-CA-H dihedral angle (θ), which is
    geometrically related to the backbone Ramachandran Phi (φ) angle.

    The relationship is empirically described by the Karplus equation:
        3J(θ) = A * cos^2(θ) - B * cos(θ) + C

    For typical L-amino acids, θ ≈ φ - 60°. This means:
    - In an ideal alpha-helix (φ ≈ -60°), θ ≈ -120°, yielding small couplings (3-5 Hz).
    - In a beta-sheet (φ ≈ -120° to -140°), θ ≈ 180°, yielding large couplings (8-10 Hz).

    EDUCATIONAL NOTE — Proline and Secondary Amines
    ===============================================
    Proline (PRO) and its D-isomer (DPR) are unique among standard amino acids
    because their sidechain cyclizes onto the backbone nitrogen, forming a
    secondary amine. Consequently, the nitrogen atom lacks an attached hydrogen
    when incorporated into a peptide bond. Without an amide proton, there is no
    H-N-CA-H spin system, and thus the 3J(HN-HA) coupling is physically undefined.
    Our shim layer explicitly strips these residues from the results.

    EDUCATIONAL NOTE — D-Amino Acids and Stereochemistry
    ====================================================
    D-amino acids are non-superimposable mirror images of natural L-amino acids.
    Because the Karplus equation is parameterized for the natural L-configuration,
    directly feeding a D-amino acid's φ angle into the standard Karplus curve
    yields incorrect results (e.g., predicting ~6.3 Hz instead of ~3.8 Hz for a
    D-alpha helix where φ ≈ +60°).

    Due to mirror symmetry, the geometric relationship inverts:
    θ_D(φ) = -θ_L(-φ). Therefore, we can accurately predict the D-amino acid
    coupling by evaluating the L-parameterized Karplus equation at the
    negated phi angle: J_D(φ) = J_L(-φ).

    Args:
        structure: Biotite AtomArray

    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz).
    """
    raw_couplings = _j.calculate_hn_ha_coupling(structure)
    filtered_couplings: typing.Dict[str, typing.Dict[int, float]] = {}

    proline_names = {"PRO", "DPR"}
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

    # Pre-calculate phi map for D-amino acid correction
    phi, _, _ = struc.dihedral_backbone(structure)
    chain_ids, res_ids, _, _ = get_residue_info(structure)
    phi_map = {}
    for c, r, p in zip(chain_ids, res_ids, phi):
        phi_map[(c, int(r))] = np.degrees(p)

    for chain_id, res_dict in raw_couplings.items():
        filtered_chain = {}
        for res_id, j_val in res_dict.items():
            # Check residue name from structure
            res_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
            if not res_mask.any():
                continue
            res_name = structure.res_name[res_mask][0]

            # 1. Proline has no HN proton; skip
            if res_name in proline_names:
                continue

            # 2. D-amino acids require stereochemical inversion for Karplus evaluation
            if res_name in d_amino_acids:
                p_angle = phi_map.get((chain_id, res_id))
                if p_angle is not None and not np.isnan(p_angle):
                    # J_D(phi) = J_L(-phi) due to stereochemical inversion
                    j_val = float(round(_j.calculate_hn_ha_coupling_from_phi(-p_angle), 2))

            filtered_chain[res_id] = j_val

        if filtered_chain:
            filtered_couplings[chain_id] = filtered_chain

    return filtered_couplings


__all__ = [
    "calculate_hn_ha_coupling",
    "predict_couplings_from_structure",
]
