import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM
from synth_pdb.validator import PDBValidator


@pytest.mark.skipif(not HAS_OPENMM, reason="PTM robustness tests require OpenMM physics engine")
class TestPTMMutationRobustness:
    """
    Stress-tests for combinations of PTMs, D-Amino Acids, and Terminal Caps.
    Ensures that residue renaming, atom stripping, and re-indexing are robust.
    """

    @pytest.mark.parametrize("ptm", ["SEP", "TPO", "PTR"])
    @pytest.mark.parametrize("cap", [True, False])
    def test_ptm_at_termini_with_capping(self, ptm: str, cap: bool) -> None:
        """
        Verify that PTMs at N or C terminals don't conflict with ACE/NME caps.
        """
        # Test N-term PTM
        seq_n = f"{ptm}-ALA-ALA"
        pdb_n = generate_pdb_content(sequence_str=seq_n, cap_termini=cap, minimize_energy=True)

        assert ptm in pdb_n
        if cap:
            assert "ACE" in pdb_n
            # Check residue IDs are unique
            lines = [line for line in pdb_n.splitlines() if line.startswith(("ATOM", "HETATM"))]
            res_ids = [int(line[22:26].strip()) for line in lines]
            # Use count check - set of res IDs should match count of unique residue names/numbers
            unique_res = {(line[17:20], line[22:26]) for line in lines}
            assert len(set(res_ids)) == len(unique_res)

        # Test C-term PTM
        seq_c = f"ALA-ALA-{ptm}"
        pdb_c = generate_pdb_content(sequence_str=seq_c, cap_termini=cap, minimize_energy=True)
        assert ptm in pdb_c
        if cap:
            assert "NME" in pdb_c

    def test_d_amino_and_ptm_combination(self) -> None:
        """
        Combination of D-amino acids and Phosphorylation in the same chain.
        """
        sequence = "D-ALA-SEP-D-TRP-TPO"
        pdb_content = generate_pdb_content(
            sequence_str=sequence, minimize_energy=True, cap_termini=True
        )

        assert "DAL" in pdb_content
        assert "SEP" in pdb_content
        assert "DTR" in pdb_content
        assert "TPO" in pdb_content
        assert "ACE" in pdb_content
        assert "NME" in pdb_content

        # Verify chirality of DAL (Res 2 because of ACE cap)
        validator = PDBValidator(pdb_content)
        ats = validator.atoms

        def get_c(res_num: int, name: str) -> np.ndarray:
            return [
                a["coords"]
                for a in ats
                if a["residue_number"] == res_num and a["atom_name"] == name
            ][0]

        # Improper N-CA-C-CB for DAL at Res 2
        imp_dal = validator._calculate_dihedral_angle(
            get_c(2, "N"), get_c(2, "CA"), get_c(2, "C"), get_c(2, "CB")
        )
        assert abs(imp_dal) > 20

    def test_all_d_amino_sweep(self) -> None:
        """
        Test that every supported D-amino acid can be generated and minimized.
        """
        from synth_pdb.data import L_TO_D_MAPPING

        for l_aa, d_aa in L_TO_D_MAPPING.items():
            if l_aa == "GLY":
                continue  # Glycine is achiral

            # Simple tripeptide with D-amino in middle
            seq = f"ALA-D-{l_aa}-ALA"
            try:
                pdb_content = generate_pdb_content(sequence_str=seq, minimize_energy=True)
                assert d_aa in pdb_content
            except Exception as e:
                pytest.fail(f"Failed to generate/minimize D-{l_aa}: {e}")

    def test_cyclic_ptm_robustness(self) -> None:
        """
        Test PTMs in a cyclic peptide.
        """
        sequence = "CYS-SEP-GLY-CYS"
        # 1-4 disulfide + cyclic
        pdb_content = generate_pdb_content(sequence_str=sequence, cyclic=True, minimize_energy=True)

        assert "SEP" in pdb_content
        assert "CONECT" in pdb_content  # Ring closure
        assert "SSBOND" in pdb_content  # Disulfide

    def test_extremely_short_ptm_cap(self) -> None:
        """
        Test a single PTM residue with both caps.
        ACE-SEP-NME
        """
        sequence = "SEP"
        pdb_content = generate_pdb_content(
            sequence_str=sequence, cap_termini=True, minimize_energy=True
        )

        assert "ACE" in pdb_content
        assert "SEP" in pdb_content
        assert "NME" in pdb_content

        # Verify 3 distinct residue IDs (1, 2, 3)
        lines = [line for line in pdb_content.splitlines() if line.startswith(("ATOM", "HETATM"))]
        res_ids = sorted({int(line[22:26].strip()) for line in lines})
        assert res_ids == [1, 2, 3]
