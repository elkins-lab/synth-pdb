import io

import biotite.structure.io.pdb as pdb
import pytest

from synth_pdb.chemical_shifts import (
    calculate_csi,
    get_secondary_structure,
    predict_chemical_shifts,
)
from synth_pdb.generator import generate_pdb_content


def test_d_peptide_csi_helix() -> None:
    """
    Test that a D-peptide helix is correctly identified by CSI deviations and SS labels.
    """
    length = 15
    # Sequence of 15 D-Alanines
    sequence = "D-ALA-" * (length - 1) + "D-ALA"

    # D-alpha helix angles (mirrored from L-alpha helix)
    # L-alpha: phi=-57, psi=-47
    # D-alpha: phi=+57, psi=+47
    phi_list = [57.0] * length
    psi_list = [47.0] * length

    pdb_content = generate_pdb_content(
        sequence_str=sequence, phi_list=phi_list, psi_list=psi_list, minimize_energy=True
    )

    f = io.StringIO(pdb_content)
    struc = pdb.PDBFile.read(f).get_structure(model=1)

    # 1. Predict chemical shifts (DAL residues)
    shifts = predict_chemical_shifts(struc, use_shiftx2=False)

    # 2. Get CSI (Chemical Shift Index) deviations
    # Wrapper should handle DAL -> ALA mapping automatically
    csi_deviations = calculate_csi(shifts, struc)

    # CA Deviations for chain A
    ca_deviations = [csi_deviations["A"].get(i + 1, 0) for i in range(length)]

    # Helix: CA Deviation > +0.7 ppm
    is_helix_by_delta = [d > 0.7 for d in ca_deviations]
    h_count_delta = is_helix_by_delta.count(True)

    # Validation: Bulk of the helix should be identified
    assert h_count_delta >= 10, f"Expected helix CA CSI > 0.7, got {h_count_delta}"

    # 3. Get categorical labels
    ss_labels = get_secondary_structure(shifts, struc)

    # Should contain 'alpha' for helix residues
    h_count_labels = ss_labels.count("alpha")
    assert h_count_labels >= 8, f"Expected at least 8 'alpha' labels, got {h_count_labels}"


@pytest.mark.xfail(
    reason="Empirical predictor lacks D-sheet (phi=135, psi=-135) training data; returns random coil."
)
def test_d_peptide_sheet() -> None:
    """
    Test that a D-peptide beta-sheet is correctly identified by CSI deviations.
    L-beta: phi=-135, psi=135
    D-beta: phi=135, psi=-135
    """
    length = 10
    sequence = "D-ALA-" * (length - 1) + "D-ALA"

    phi_list = [135.0] * length
    psi_list = [-135.0] * length

    pdb_content = generate_pdb_content(
        sequence_str=sequence, phi_list=phi_list, psi_list=psi_list, minimize_energy=True
    )

    f = io.StringIO(pdb_content)
    struc = pdb.PDBFile.read(f).get_structure(model=1)

    shifts = predict_chemical_shifts(struc, use_shiftx2=False)
    csi_deviations = calculate_csi(shifts, struc)

    ca_deviations = [csi_deviations["A"].get(i + 1, 0) for i in range(length)]

    # Beta Sheet: CA Deviation < -0.7 ppm
    is_sheet = [d < -0.7 for d in ca_deviations]
    e_count = is_sheet.count(True)

    assert (
        e_count >= 5
    ), f"Expected at least 5 residues to have CA CSI < -0.7 for sheet, got {e_count}"

    ss_labels = get_secondary_structure(shifts, struc)
    assert "beta" in ss_labels, f"Expected 'beta' in secondary structure labels, got {ss_labels}"
