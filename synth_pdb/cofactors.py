"""Cofactor and Metal Ion Coordination Module.

THE "AI TRINITY" PHASE 15: Inorganic Coordination.

Biological proteins aren't just chains of amino acids; they often require
inorganic "cofactors" or metal ions to function. Zinc (Zn2+) is one of
the most common, found in "Zinc Finger" motifs where it stabilizes the
fold by coordinating with Cysteine (C) and Histidine (H) residues.

Educational Note - Coordination Chemistry:
------------------------------------------
1. Ligands: The atoms that donate electrons to the metal (e.g., Cys Sulfur, His Nitrogen).
2. Coordination Number: The number of ligands (Zinc is usually 4 - Tetrahedral).
3. Geometric Centroid: The ideal position for the metal is the center of its ligands.

This module automatically detects these motifs and inserts the appropriate ions.
"""

import logging
from typing import Dict, List

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def find_metal_binding_sites(
    structure: struc.AtomArray, distance_threshold: float = 20.0
) -> List[Dict]:
    """Scans the structure for clusters of residues that could coordinate a metal ion.

    Args:
        structure: Biotite AtomArray.
        distance_threshold: Max distance between any two coordinating atoms in a cluster.

    Returns:
        List[Dict]: A list of detected sites, each with 'type' and 'ligand_indices'.

    """
    logger.info("Scanning for metal binding motifs (Zinc Fingers)...")

    # Standard ligands for Zinc
    # CYS (SG), HIS (NE2 or ND1) - including HID/HIE/HIP variants
    ligand_mask = ((structure.res_name == "CYS") & (structure.atom_name == "SG")) | (
        (np.isin(structure.res_name, ["HIS", "HID", "HIE", "HIP"]))
        & ((structure.atom_name == "NE2") | (structure.atom_name == "ND1"))
    )

    candidate_indices = np.where(ligand_mask)[0]
    if len(candidate_indices) < 3:
        return []

    candidate_coords = structure.coord[candidate_indices]
    sites = []
    assigned_indices = set()

    # ── Iterative Motif Detection (Tightest-Cluster First) ──────────────────
    # Unlike a simple greedy search which might pick arbitrary nearby atoms,
    # we iteratively identify the "best" coordination sites.
    #
    # The algorithm:
    # 1. For each unassigned ligand, find all neighbors within threshold.
    # 2. If 4+ ligands are found, calculate the 'spread' (average pair-wise
    #    distance) of the tightest sub-group of 4.
    # 3. Identify the cluster with the global minimum spread across the whole protein.
    # 4. 'Commit' that cluster as a site and mark atoms as assigned.
    # 5. Repeat until no more 4-ligand clusters can be formed.
    while True:
        best_cluster = None
        min_spread = float("inf")

        # Outer loop: Evaluate every possible ligand as a potential cluster 'seed'
        for i in range(len(candidate_indices)):
            idx_i = candidate_indices[i]
            # Skip if this atom was already claimed by a previous site
            if idx_i in assigned_indices:
                continue

            # Calculate distances from this seed to all other candidate ligands
            diffs = candidate_coords - candidate_coords[i]
            dists = np.sqrt(np.sum(diffs**2, axis=-1))

            # Filter unassigned neighbors that fall within the threshold
            neighbor_mask = dists < distance_threshold
            unassigned_neighbor_indices = [
                j
                for j in range(len(candidate_indices))
                if neighbor_mask[j] and candidate_indices[j] not in assigned_indices
            ]

            # Zinc Fingers traditionally coordinate with exactly 4 ligands (Cys4 or Cys2His2)
            if len(unassigned_neighbor_indices) >= 4:
                # EDUCATIONAL NOTE: Local Optimization
                # We sort neighbors by distance and take the 4 closest to the seed
                sorted_neighbor_indices = sorted(
                    unassigned_neighbor_indices, key=lambda x: float(dists[x])
                )
                cluster_indices = sorted_neighbor_indices[:4]

                # EDUCATIONAL NOTE: Geometric Realism
                # We calculate 'spread' as the average distance between ALL atoms in the group.
                # This prevents picking a string of atoms that happen to be near the threshold
                # but are far from each other.
                cluster_coords = candidate_coords[cluster_indices]
                spread = 0.0
                count = 0
                for ci in range(4):
                    for cj in range(ci + 1, 4):
                        spread += float(np.linalg.norm(cluster_coords[ci] - cluster_coords[cj]))
                        count += 1
                avg_spread = float(spread / count)

                # Track the tightest cluster found in this iteration
                if avg_spread < min_spread:
                    min_spread = avg_spread
                    best_cluster = [candidate_indices[idx] for idx in cluster_indices]

        # If a valid cluster was found, register it and continue scanning
        if best_cluster:
            sites.append({"type": "ZN", "ligand_indices": best_cluster})
            for idx in best_cluster:
                assigned_indices.add(idx)
        else:
            # Termination condition: No more unassigned groups of 4 found
            break

    if sites:
        logger.info(f"Found {len(sites)} potential metal binding sites.")
    return sites


def add_metal_ion(structure: struc.AtomArray, site: Dict) -> struc.AtomArray:
    """Adds a metal ion (HETATM) to the structure at the centroid of its ligands.

    Args:
        structure: Original AtomArray.
        site: Identification of the site (from find_metal_binding_sites).

    Returns:
        struc.AtomArray: New AtomArray with the ion appended.

    """
    ion_type = site["type"]
    ligand_indices = site["ligand_indices"]

    # Calculate Centroid
    ligand_coords = structure.coord[ligand_indices]
    centroid = np.mean(ligand_coords, axis=0)

    # Create the Ion Atom
    ion = struc.AtomArray(1)
    ion.res_name = np.array([ion_type])
    ion.atom_name = np.array([ion_type])
    ion.element = np.array([ion_type])  # "ZN" not "Z" for Zinc
    ion.coord = np.array([centroid])

    # Metadata
    # Pick a residue ID higher than existing
    max_res_id = np.max(structure.res_id)
    ion.res_id = np.array([max_res_id + 1])
    ion.chain_id = np.array([structure.chain_id[0]])
    ion.hetero = np.array([True])

    logger.info(f"Injected {ion_type} ion at coordinated site {centroid}.")

    return structure + ion
