import os
import tempfile

from synth_pdb.analysis import GeometryAnalyzer
from synth_pdb.generator import generate_pdb_content


def test_compare_pdbs():
    """Test RMSD and superposition alignment between two PDB files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate two slightly different PDBs
        pdb1_path = os.path.join(tmpdir, "pdb1.pdb")
        pdb2_path = os.path.join(tmpdir, "pdb2.pdb")

        # Ref: Alpha-helix
        content1 = generate_pdb_content(sequence_str="AAAAA", conformation="alpha", seed=42)
        with open(pdb1_path, "w") as f:
            f.write(content1)

        # Ref: Alpha-helix with some drift
        content2 = generate_pdb_content(sequence_str="AAAAA", conformation="alpha", seed=42, drift=2.0)
        with open(pdb2_path, "w") as f:
            f.write(content2)

        analyzer = GeometryAnalyzer()
        result = analyzer.compare_pdbs(pdb1_path, pdb2_path, ca_only=True)

        assert "rmsd" in result
        assert "rotation" in result
        assert "translation" in result
        assert result["rmsd"] > 0.0  # Drift should lead to some RMSD
        assert result["rotation"].shape == (3, 3)
        assert result["translation"].shape == (3,)

def test_analyze_ensemble_pdbs():
    """Test batch analysis using a list of PDB paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            path = os.path.join(tmpdir, f"decoy_{i}.pdb")
            content = generate_pdb_content(sequence_str="AAAAA", conformation="alpha", seed=i)
            with open(path, "w") as f:
                f.write(content)
            paths.append(path)

        analyzer = GeometryAnalyzer()
        result = analyzer.analyze_ensemble_pdbs(paths)

        assert "avg_rmsd" in result
        assert "medoid_index" in result
        assert "medoid_path" in result
        assert 0 <= result["medoid_index"] < 3
        assert result["medoid_path"] in paths

def test_calculate_residue_strain():
    """Test detection of non-planar peptide bonds (Omega outliers)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "strained.pdb")

        # We'll "strain" it by generating with high drift or manual edit?
        # Let's generate a normal one first.
        content = generate_pdb_content(sequence_str="AAAAA", conformation="alpha", seed=42)

        # Manual edit to make a "cis" peptide bond (strain)
        # Omega is 180 (trans) usually. Let's make it 0.
        # This is complex to edit manually. Let's just use high drift.
        # Actually, let's use the 'omega_list' parameter in generate_pdb_content (if it exists)
        # to force a cis-peptide bond.

        # Check if generate_pdb_content accepts omega_list (from generator.py view it does)
        strained_content = generate_pdb_content(
            sequence_str="AAAAA",
            conformation="alpha",
            omega_list=[0.0, 180.0, 180.0, 180.0, 180.0]
        )

        with open(path, "w") as f:
            f.write(strained_content)

        analyzer = GeometryAnalyzer()
        strain_map = analyzer.calculate_residue_strain(path)

        # Residue 1 omega should be near 0, so deviation = |0 - 180| = 180.
        # res_ids are usually 1, 2, 3...
        assert 1 in strain_map
        assert strain_map[1] > 170.0 # High strain for the forced cis-bond
