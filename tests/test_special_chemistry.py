from typing import Any

import biotite.structure as struc
import numpy as np

from synth_pdb.special_chemistry import find_gfp_chromophore_motif, form_gfp_chromophore


def create_mock_structure(res_names: list[str]) -> struc.AtomArray:
    """Helper to create a simple AtomArray for testing."""
    n_residues = len(res_names)
    structure = struc.AtomArray(n_residues)
    structure.res_name = np.array(res_names)
    structure.res_id = np.arange(1, n_residues + 1)
    structure.chain_id = np.array(["A"] * n_residues)
    return structure


def test_find_gfp_motif_present() -> None:
    """Test that the SYG motif is found when present."""
    structure = create_mock_structure(["LEU", "SER", "TYR", "GLY", "PHE"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is not None
    assert motif["ser_res_id"] == 2
    assert motif["tyr_res_id"] == 3
    assert motif["gly_res_id"] == 4


def test_find_gfp_motif_not_present() -> None:
    """Test that the SYG motif is not found when absent."""
    structure = create_mock_structure(["LEU", "SER", "ALA", "GLY", "PHE"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is None


def test_find_gfp_motif_at_end() -> None:
    """Test that the SYG motif is found at the end of the chain."""
    structure = create_mock_structure(["LEU", "PHE", "SER", "TYR", "GLY"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is not None
    assert motif["ser_res_id"] == 3


def test_find_gfp_motif_multiple_chains() -> None:
    """Test that the function returns None for multiple chains."""
    structure = create_mock_structure(["SER", "TYR", "GLY"])
    structure.chain_id = np.array(["A", "B", "C"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is None


def test_form_gfp_chromophore_implementation(caplog: Any) -> None:
    """Test that the function renames residues to CRO and logs success."""
    structure = create_mock_structure(["SER", "TYR", "GLY"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is not None

    with caplog.at_level("INFO"):
        returned_structure = form_gfp_chromophore(structure, motif)
        assert "Successfully matured SYG motif" in caplog.text
        assert "into CRO" in caplog.text
        # Check that residues are renamed
        assert np.all(returned_structure.res_name == "CRO")
        # Ensure it returned a copy/modified version, not the exact same object if we want to be strict
        # (though implementation uses .copy() so this is true)
        assert returned_structure is not structure


def test_find_gfp_motif_partial_match() -> None:
    """Test that partial matches (e.g. S-A-G) are NOT found."""
    structure = create_mock_structure(["SER", "ALA", "GLY"])
    motif = find_gfp_chromophore_motif(structure)
    assert motif is None


def test_find_gfp_motif_chain_boundary() -> None:
    """Test that motif is not found if split across chain boundary (handled by multiple chains test usually)."""
    # Create a structure that looks like SER-TYR-GLY but has different chain IDs
    structure = create_mock_structure(["SER", "TYR", "GLY"])
    structure.chain_id = np.array(["A", "A", "B"])
    motif = find_gfp_chromophore_motif(structure)
    # Current implementation checks for unique chain_id, so this should fail
    assert motif is None
