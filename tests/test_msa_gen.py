import biotite.structure as struc
import numpy as np

from synth_pdb import evolution


def create_mock_structure():
    """Creates a simple structure (Alpha Helix like)."""
    # We create a dummy atom array with coords.
    # 3 residues: ALA, ALA, ALA
    atoms = struc.AtomArray(3)
    atoms.res_name = np.array(["ALA", "LEU", "GLY"])
    atoms.res_id = np.array([1, 2, 3])
    atoms.element = np.array(["C", "C", "C"])
    # Add dummy coordinates for SASA
    atoms.coord = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]])
    return atoms


class TestMSAGenerator:

    def test_sasa_identification(self, mocker):
        """Test detection of buried vs exposed residues."""
        atoms = create_mock_structure()

        # Mock biotite.structure.sasa
        mock_areas = np.array([100.0, 10.0, 100.0])  # Res 1, 2, 3
        mocker.patch("biotite.structure.sasa", return_value=mock_areas)

        # Mock apply_residue_wise to just return the areas
        mocker.patch("biotite.structure.apply_residue_wise", return_value=mock_areas)

        # Mock get_residue_starts
        mocker.patch("biotite.structure.get_residue_starts", return_value=[0, 1, 2])

        # Run function
        rel_sasa = evolution.calculate_relative_sasa(atoms)

        assert len(rel_sasa) == 3
        assert rel_sasa[0] > 0.5  # Exposed
        assert rel_sasa[1] < 0.2  # Buried

    def test_msa_generation_conservation(self, mocker):
        """Verify Core residues mutate less or conservatively."""
        atoms = create_mock_structure()

        # Mock SASA: Middle residue (L) is BURIED (0% relative SASA)
        mock_rel_sasa = np.array([1.0, 0.0, 1.0])
        mocker.patch("synth_pdb.evolution.calculate_relative_sasa", return_value=mock_rel_sasa)

        # Generate raw sequences
        msa_seqs = evolution.generate_msa_sequences(atoms, n_seqs=100, mutation_rate=0.5)

        # Pos 1 (L -> Buried) should be L or conserved hydrophobic (V, I, F)
        pos1_variants = [s[1] for s in msa_seqs]
        charged = set("DEKR")

        # Count non-hydrophobic mutations in core
        violations = sum(1 for aa in pos1_variants if aa in charged)
        assert violations < 5, "Buried Core residue mutated to charged residue too often!"

        # Check Pos 0 variation
        pos0_variants = {s[0] for s in msa_seqs}
        assert len(pos0_variants) > 1, "Exposed residue did not mutate at all with high rate"

    def test_write_msa(self, tmp_path):
        """Test FASTA export."""
        sequences = ["AAA", "AAB", "AAC"]
        outfile = tmp_path / "test.fasta"

        evolution.write_msa(sequences, str(outfile))

        content = outfile.read_text()
        assert ">seq_0" in content
        assert "AAA" in content
