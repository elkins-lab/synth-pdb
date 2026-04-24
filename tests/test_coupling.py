import io

import biotite.structure.io.pdb as pdb
import numpy as np

from synth_pdb.coupling import calculate_hn_ha_coupling, predict_couplings_from_structure
from synth_pdb.generator import generate_pdb_content


class TestJCoupling:
    """TDD Test Suite for J-Coupling Prediction.

    Verifies the implementation of the Karplus Equation for 3J(HN-HA) couplings.
    """

    def test_ideal_alpha_helix(self) -> None:
        """Test J-coupling for an ideal Alpha Helix.
        Phi approx -57 degrees.
        Expected J: Small (< 6.0 Hz).
        """
        phi = -57.0
        j_val = calculate_hn_ha_coupling(phi)

        # Textbook expectation: ~3.9 - 5.0 Hz
        assert 2.0 < j_val < 6.0, f"Helix coupling {j_val} out of expected range (2-6 Hz)"

    def test_ideal_beta_sheet(self) -> None:
        """Test J-coupling for an ideal Beta Sheet.
        Phi approx -139 degrees (parallel) / -119 (antiparallel).
        Expected J: Large (> 8.0 Hz).
        """
        # Beta sheet average ~ -120 to -140
        phi = -120.0
        j_val = calculate_hn_ha_coupling(phi)

        # Textbook expectation: ~8.0 - 10.0 Hz
        assert j_val > 7.5, f"Beta sheet coupling {j_val} too small (expected > 7.5 Hz)"

    def test_periodicity(self) -> None:
        """Karplus equation should be periodic (360 degrees)."""
        j1 = calculate_hn_ha_coupling(-60.0)
        j2 = calculate_hn_ha_coupling(300.0)  # -60 + 360
        assert np.isclose(j1, j2), "Karplus function is not periodic!"

    def test_predict_alpha_helix_structure(self) -> None:
        """Test J-coupling prediction from an actual alpha-helix structure."""
        length = 10
        pdb_content = generate_pdb_content(
            sequence_str="ALA-" * (length - 1) + "ALA",
            phi_list=[-57.0] * length,
            psi_list=[-47.0] * length,
            minimize_energy=False,
        )
        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        couplings = predict_couplings_from_structure(struc)
        assert "A" in couplings
        chain_couplings = couplings["A"]

        # Check central residues
        for res_id in range(2, length):
            assert 2.0 < chain_couplings[res_id] < 6.0

    def test_predict_beta_sheet_structure(self) -> None:
        """Test J-coupling prediction from an actual beta-sheet structure."""
        length = 10
        pdb_content = generate_pdb_content(
            sequence_str="ALA-" * (length - 1) + "ALA",
            phi_list=[-135.0] * length,
            psi_list=[135.0] * length,
            minimize_energy=False,
        )
        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        couplings = predict_couplings_from_structure(struc)
        assert "A" in couplings
        chain_couplings = couplings["A"]

        # Check central residues
        for res_id in range(2, length):
            assert chain_couplings[res_id] > 7.5

    def test_d_peptide_coupling(self) -> None:
        """Test J-coupling prediction from a D-amino acid peptide structure."""
        length = 5
        pdb_content = generate_pdb_content(
            sequence_str="D-ALA-" * (length - 1) + "D-ALA",
            phi_list=[57.0] * length,
            psi_list=[47.0] * length,
            minimize_energy=False,
        )
        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        couplings = predict_couplings_from_structure(struc)
        assert "A" in couplings
        chain_couplings = couplings["A"]

        for res_id in range(2, length):
            # Same magnitude as L-helix because Karplus cos(theta) is symmetric
            assert 2.0 < chain_couplings[res_id] < 6.0

    def test_proline_coupling_skipped(self) -> None:
        """Proline has no amide proton, so its coupling must be omitted."""
        pdb_content = generate_pdb_content(sequence_str="ALA-PRO-ALA", minimize_energy=False)
        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        couplings = predict_couplings_from_structure(struc)
        assert "A" in couplings
        chain_couplings = couplings["A"]

        # Proline is residue 2, should be excluded
        assert 2 not in chain_couplings
