import os
import tempfile
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.analysis import GeometryAnalyzer
from synth_pdb.generator import generate_pdb_content


def test_compare_pdbs_mismatch() -> None:
    """Test comparison with atom count mismatch."""
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "p1.pdb")
        p2 = os.path.join(tmp, "p2.pdb")

        with open(p1, "w") as f:
            f.write(generate_pdb_content(length=5))
        with open(p2, "w") as f:
            f.write(generate_pdb_content(length=10))

        with pytest.raises(ValueError, match="Atom count mismatch"):
            GeometryAnalyzer.compare_pdbs(p1, p2)


def test_compare_pdbs_happy_path() -> None:
    """Test successful comparison of two identical PDBs."""
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "p1.pdb")
        content = generate_pdb_content(length=5, sequence_str="AAAAA")
        with open(p1, "w") as f:
            f.write(content)

        results = GeometryAnalyzer.compare_pdbs(p1, p1)
        assert results["rmsd"] < 1e-4
        assert results["rotation"].shape == (3, 3)


def test_analyze_ensemble_pdbs() -> None:
    """Test ensemble analysis from list of PDB paths."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f"p{i}.pdb")
            with open(p, "w") as f:
                # Use same seed for identical, or drift for varied
                f.write(generate_pdb_content(length=5, seed=i))
            paths.append(p)

        results = GeometryAnalyzer.analyze_ensemble_pdbs(paths)
        assert "avg_rmsd" in results
        assert "medoid_index" in results
        assert results["medoid_path"] in paths


def test_calculate_residue_strain() -> None:
    """Test residue strain (omega deviation) calculation."""
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "p1.pdb")
        # Alpha helix has trans (180) peptide bonds
        content = generate_pdb_content(length=5, conformation="alpha")
        with open(p1, "w") as f:
            f.write(content)

        strain = GeometryAnalyzer.calculate_residue_strain(p1)
        # Should have strain for L-1 residues
        assert len(strain) == 4
        for val in strain.values():
            # Standard generation is close to trans, but allowed small deviations
            assert val < 15.0


def test_rmsd_to_average() -> None:
    """Test low-level rmsd_to_average utility."""
    from synth_pdb.geometry.rmsd import calculate_rmsd_to_average

    c1 = np.array([[0, 0, 0], [1, 1, 1]])
    c2 = np.array([[0, 0, 0], [1.1, 1.1, 1.1]])

    avg_rmsd, avg_coords = calculate_rmsd_to_average([c1, c2])
    assert avg_rmsd > 0
    assert avg_coords.shape == (2, 3)
