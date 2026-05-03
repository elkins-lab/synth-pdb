#!/usr/bin/env python3
"""
Publication Figure Generator.

This script demonstrates the high-fidelity plotting capabilities of synth-pdb
by generating journal-ready figures for Human Ubiquitin (1D3Z).
"""

import os
import sys
import numpy as np
import biotite.structure.io.pdb as pdb_io
import biotite.structure as struc

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth_pdb.bmrb_api import BMRBAPI
from synth_pdb.chemical_shifts import predict_chemical_shifts
from synth_pdb.saxs import calculate_saxs_profile, calculate_radius_of_gyration
from synth_pdb.quality.plots import (
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
)


def main():
    output_dir = "artifacts/publication_figures"
    os.makedirs(output_dir, exist_ok=True)

    pdb_id = "1D3Z"
    bmrb_id = "6457"
    pdb_path = f"examples/{pdb_id}.pdb"

    print(f"--- Generating Publication Figures for {pdb_id} ---")

    # 1. Load Structure
    if not os.path.exists(pdb_path):
        BMRBAPI.download_pdb(pdb_id, pdb_path)

    pdb_file = pdb_io.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)

    # 2. Ramachandran Plot
    print("Generating Ramachandran plot...")
    # Calculate dihedrals
    phi, psi, omega = struc.dihedral_backbone(structure)
    # Filter NaN (termini)
    mask = ~np.isnan(phi) & ~np.isnan(psi)
    phi_deg = np.degrees(phi[mask])
    psi_deg = np.degrees(psi[mask])

    plot_ramachandran_publication(
        phi_deg,
        psi_deg,
        title=f"Ramachandran: {pdb_id}",
        output_path=os.path.join(output_dir, "figure_1_ramachandran.pdf"),
    )

    # 3. Chemical Shift Correlation (CA)
    print("Generating Chemical Shift correlation...")
    exp_shifts = BMRBAPI.fetch_chemical_shifts(bmrb_id)
    syn_shifts_full = predict_chemical_shifts(structure)
    syn_shifts = syn_shifts_full.get("A", list(syn_shifts_full.values())[0])

    plot_chemical_shift_correlation(
        exp_shifts,
        syn_shifts,
        atom_type="CA",
        output_path=os.path.join(output_dir, "figure_2_cs_correlation.pdf"),
    )

    # 4. SAXS Profile
    print("Generating SAXS profile...")
    q, intensity = calculate_saxs_profile(structure, q_max=0.3)
    rg = calculate_radius_of_gyration(structure)

    plot_saxs_publication(
        q, intensity, rg=rg, output_path=os.path.join(output_dir, "figure_3_saxs.pdf")
    )

    print(f"\nSuccess! Publication-ready figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
