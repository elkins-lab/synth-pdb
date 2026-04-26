"""
Sidechain reconstruction and rotation utilities.
"""

import logging

import biotite.structure as struc
import numpy as np

from synth_pdb.geometry._numba import njit

logger = logging.getLogger(__name__)


@njit
def rotate_points(
    points: np.ndarray, axis_p1: np.ndarray, axis_p2: np.ndarray, angle_deg: float
) -> np.ndarray:
    """Rotate points about an axis defined by two points."""
    # Translate to origin
    v = (axis_p2 - axis_p1).astype(np.float64)
    v_norm = np.sqrt(np.sum(v**2))
    if v_norm > 0:
        v = v / v_norm

    # Rodriguez rotation formula or similar
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    # Pre-allocate for JIT efficiency and type stability
    n_points = points.shape[0]
    result = np.zeros((n_points, 3), dtype=np.float64)

    for i in range(n_points):
        p = points[i]
        px = p.astype(np.float64) - axis_p1.astype(np.float64)

        # Project px onto v
        proj = np.sum(px * v) * v
        perp = px - proj

        # Rotate perp component
        w = np.cross(v, perp)
        rotated_perp = perp * c + w * s

        result[i] = proj + rotated_perp + axis_p1.astype(np.float64)
    return result


def reconstruct_sidechain(
    peptide: struc.AtomArray,
    res_id: int,
    rotamer: dict[str, float | list[float]],
    res_name: str | None = None,
) -> None:
    """Reconstructs the sidechain of a specific residue in the peptide using the provided rotamer angles.
    Updates the coordinates in place.
    """
    # 1. Isolate the residue atoms
    mask = peptide.res_id == res_id
    if not np.any(mask):
        raise ValueError(f"Residue {res_id} not found in structure.")

    res_atoms_indices = np.where(mask)[0]

    # Get backbone atoms as reference
    try:
        n_idx = np.where((peptide.res_id == res_id) & (peptide.atom_name == "N"))[0][0]
        ca_idx = np.where((peptide.res_id == res_id) & (peptide.atom_name == "CA"))[0][0]
        c_idx = np.where((peptide.res_id == res_id) & (peptide.atom_name == "C"))[0][0]
    except IndexError:
        logger.warning(
            f"Residue {res_id} missing backbone atoms (N, CA, C). Cannot reconstruct sidechain."
        )
        return

    if res_name is None:
        res_name = peptide.res_name[ca_idx]

    # Get standard template
    try:
        ref_res_template = struc.info.residue(res_name).copy()
    except KeyError:
        logger.warning(f"Unknown residue {res_name}, cannot reconstruct.")
        return

    if "chi1" not in rotamer:
        return

    # Basic rigid body alignment of the whole template to the backbone

    mob_n = ref_res_template[ref_res_template.atom_name == "N"]
    mob_ca = ref_res_template[ref_res_template.atom_name == "CA"]
    mob_c = ref_res_template[ref_res_template.atom_name == "C"]

    if len(mob_n) == 0 or len(mob_ca) == 0 or len(mob_c) == 0:
        return

    mobile_bb = struc.array([mob_n[0], mob_ca[0], mob_c[0]])
    target_bb = struc.array([peptide[n_idx], peptide[ca_idx], peptide[c_idx]])

    _, transformation = struc.superimpose(mobile_bb, target_bb)
    ref_res_template.coord = transformation.apply(ref_res_template.coord)

    # 2. Apply Chi1 rotation
    if "chi1" in rotamer and len(ref_res_template[ref_res_template.atom_name == "CB"]) > 0:
        ca_atom = ref_res_template[ref_res_template.atom_name == "CA"][0]
        cb_atom = ref_res_template[ref_res_template.atom_name == "CB"][0]

        backbone_mask = np.isin(ref_res_template.atom_name, ["N", "CA", "C", "O", "H", "HA", "CB"])
        sidechain_mask = ~backbone_mask

        if np.any(sidechain_mask):
            gamma_atoms = [
                a
                for a in ref_res_template
                if a.atom_name.startswith("CG")
                or a.atom_name.startswith("OG")
                or a.atom_name.startswith("SG")
            ]
            if gamma_atoms:
                g_atom = gamma_atoms[0]
                n_atom = ref_res_template[ref_res_template.atom_name == "N"][0]

                curr_chi1 = struc.dihedral(n_atom.coord, ca_atom.coord, cb_atom.coord, g_atom.coord)
                curr_chi1_deg = np.rad2deg(curr_chi1)

                _chi1_val_r = rotamer["chi1"]
                target_chi1 = (
                    _chi1_val_r[0] if isinstance(_chi1_val_r, list) else float(_chi1_val_r)
                )
                delta_deg = target_chi1 - curr_chi1_deg

                sidechain_indices = np.where(sidechain_mask)[0]
                coords_to_rot = ref_res_template.coord[sidechain_indices]

                rotated_coords = rotate_points(
                    coords_to_rot, ca_atom.coord, cb_atom.coord, delta_deg
                )
                ref_res_template.coord[sidechain_indices] = rotated_coords

    # Update original peptide coordinates
    for i in res_atoms_indices:
        atom_name = peptide.atom_name[i]
        temp_idx = np.where(ref_res_template.atom_name == atom_name)[0]
        if len(temp_idx) > 0:
            peptide.coord[i] = ref_res_template.coord[temp_idx[0]]
