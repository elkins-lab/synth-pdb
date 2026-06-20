import csv
import json
import logging
from typing import Any

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def calculate_torsion_angles(atom_array: struc.AtomArray) -> list[dict[str, Any]]:
    """Calculate backbone torsion angles (phi, psi, omega) for a given structure.

    Args:
        atom_array: Biotite AtomArray containing the protein structure.

    Returns:
        List of dictionaries, one per residue, containing:
        - residue: 3-letter code
        - res_id: Residue number
        - phi: Phi angle in degrees (or None)
        - psi: Psi angle in degrees (or None)
        - omega: Omega angle in degrees (or None)

    """
    logger.info("Calculating backbone torsion angles...")

    # Calculate angles using Biotite
    # dihedral_backbone returns (phi, psi, omega) in RADIANS for each residue.
    # Note: biotite.structure.dihedral_backbone calculates phi, psi, omega for each residue.
    # The first residue's phi is usually NaN, last residue's psi is NaN.

    phi, psi, omega = struc.dihedral_backbone(atom_array)

    # Convert to degrees
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    omega_deg = np.degrees(omega)

    # Get residue identifiers
    # We iterate through indices corresponding to the backbone results
    # Ideally should use residue iterator to be safe, but backbone arrays map to residues.

    # Get unique residue IDs to map back
    residue_starts: np.ndarray = struc.get_residue_starts(atom_array)

    results = []

    for i, start_idx in enumerate(residue_starts):
        res_name = atom_array.res_name[start_idx]
        res_id = int(atom_array.res_id[start_idx])

        # Handle NaN values (convert to None for JSON compliance)
        p = phi_deg[i]
        ps = psi_deg[i]
        o = omega_deg[i]

        if np.isnan(p):
            p = None
        if np.isnan(ps):
            ps = None
        if np.isnan(o):
            o = None

        entry = {"residue": res_name, "res_id": res_id, "phi": p, "psi": ps, "omega": o}
        results.append(entry)

    logger.info(f"Calculated angles for {len(results)} residues.")
    return results


def export_torsion_angles(data: list[dict[str, Any]], output_file: str, fmt: str = "csv") -> None:
    """Export calculated torsion angles to a file.

    Args:
        data: List of angle dictionaries (from calculate_torsion_angles).
        output_file: Path to output file.
        fmt: Format 'csv' or 'json'.

    """
    logger.info(f"Exporting torsion angles to {output_file} ({fmt})...")

    if fmt.lower() == "json":
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    elif fmt.lower() == "csv":
        fieldnames = ["residue", "res_id", "phi", "psi", "omega"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                # Ensure None becomes empty string or specific marker if needed
                # csv module handles None as empty string by default? No, usually expects string.
                # Actually DictWriter handles it if we don't convert.
                # But let's be explicit solely for robustness if needed.
                writer.writerow(row)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    logger.info("Export complete.")
