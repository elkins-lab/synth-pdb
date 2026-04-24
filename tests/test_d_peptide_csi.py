import io

import biotite.structure.io.pdb as pdb

from synth_pdb.chemical_shifts import (
    calculate_csi,
    get_secondary_structure,
    predict_chemical_shifts,
)
from synth_pdb.generator import generate_pdb_content


def test_d_peptide_csi_helix() -> None:
    """
    Test that a D-peptide helix is correctly identified by CSI deviations and SS labels.

    Because OpenMM minimization force fields are heavily parameterized for L-amino acids,
    directly generating and minimizing a D-helix (phi=57, psi=47) causes the physics engine
    to distort the geometry into an unnatural local minimum. To bypass this OpenMM artifact
    and test the empirical predictor's mathematical enantiomer logic properly, we generate
    and minimize a stable L-helix, then mathematically mirror the coordinates.
    """
    length = 15
    # Sequence of 15 L-Alanines to generate stable core
    sequence = "ALA-" * (length - 1) + "ALA"

    # L-alpha helix angles
    phi_list = [-57.0] * length
    psi_list = [-47.0] * length

    pdb_content = generate_pdb_content(
        sequence_str=sequence, phi_list=phi_list, psi_list=psi_list, minimize_energy=True
    )

    f = io.StringIO(pdb_content)
    struc_l = pdb.PDBFile.read(f).get_structure(model=1)

    # Mathematically mirror to create the D-enantiomer
    struc_d = struc_l.copy()
    struc_d.coord = -struc_d.coord

    # Rename to D-Amino acid
    mask = struc_d.res_name == "ALA"
    struc_d.res_name[mask] = "DAL"

    # 1. Predict chemical shifts (DAL residues)
    shifts = predict_chemical_shifts(struc_d, use_shiftx2=False)

    # 2. Get CSI (Chemical Shift Index) deviations
    csi_deviations = calculate_csi(shifts, struc_d)

    # CA Deviations for chain A
    ca_deviations = [csi_deviations["A"].get(i + 1, 0) for i in range(length)]

    # Helix: CA Deviation > +0.7 ppm
    is_helix_by_delta = [d > 0.7 for d in ca_deviations]
    h_count_delta = is_helix_by_delta.count(True)

    # Validation: Bulk of the helix should be identified
    assert h_count_delta >= 10, f"Expected helix CA CSI > 0.7, got {h_count_delta}"

    # 3. Get categorical labels
    ss_labels = get_secondary_structure(shifts, struc_d)

    # Should contain 'alpha' for helix residues
    h_count_labels = ss_labels.count("alpha")
    assert h_count_labels >= 8, f"Expected at least 8 'alpha' labels, got {h_count_labels}"


def test_d_peptide_sheet_parity() -> None:
    """
    Test that a D-peptide beta-sheet produces equivalent chemical shifts to an L-peptide beta-sheet.
    L-beta: phi=-135, psi=135
    D-beta: phi=135, psi=-135

    Because an isolated beta strand is unstable and will collapse during energy
    minimization, we use `minimize_energy=False` to preserve the extended geometry.
    Then we assert that the empirical predictor's D-amino acid logic correctly
    reproduces the L-peptide shifts for the target geometry.
    """
    length = 10

    # Generate L-sheet
    l_seq = "ALA-" * (length - 1) + "ALA"
    l_pdb = generate_pdb_content(
        sequence_str=l_seq,
        phi_list=[-135.0] * length,
        psi_list=[135.0] * length,
        minimize_energy=False,
    )
    f_l = io.StringIO(l_pdb)
    l_struc = pdb.PDBFile.read(f_l).get_structure(model=1)
    l_shifts = predict_chemical_shifts(l_struc, use_shiftx2=False)

    # Generate D-sheet
    d_seq = "D-ALA-" * (length - 1) + "D-ALA"
    d_pdb = generate_pdb_content(
        sequence_str=d_seq,
        phi_list=[135.0] * length,
        psi_list=[-135.0] * length,
        minimize_energy=False,
    )
    f_d = io.StringIO(d_pdb)
    d_struc = pdb.PDBFile.read(f_d).get_structure(model=1)
    d_shifts = predict_chemical_shifts(d_struc, use_shiftx2=False)

    # Compare CA shifts
    l_ca = [l_shifts["A"][i + 1]["CA"] for i in range(length)]
    d_ca = [d_shifts["A"][i + 1]["CA"] for i in range(length)]

    # Assert arrays are the same length
    assert len(l_ca) == length
    assert len(d_ca) == length

    # Validate absolute RMSD
    import numpy as np

    l_ca_arr = np.array(l_ca)
    d_ca_arr = np.array(d_ca)

    rmsd = np.sqrt(np.mean((l_ca_arr - d_ca_arr) ** 2))

    # We expect very low RMSD
    # (RMSD is not exactly 0.0 because default CH3 rotamer conformers
    # in standard Biotite generation may differ slightly between L and D templates)
    assert rmsd < 0.5, f"Expected CA shift RMSD < 0.5 ppm, got {rmsd} ppm"
