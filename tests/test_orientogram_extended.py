import numpy as np
import pytest
from synth_pdb.orientogram import compute_6d_orientations


def test_compute_6d_orientations_standard_shape() -> None:
    """Test with input matching exactly (B, L*4, 3) - N, CA, C, CB."""
    # 2 residues, 1 batch
    coords = np.random.rand(1, 8, 3)
    atom_names = ["N", "CA", "C", "CB", "N", "CA", "C", "CB"]
    residue_indices = [1, 1, 1, 1, 2, 2, 2, 2]

    orientations = compute_6d_orientations(coords, atom_names, residue_indices, 2)

    assert orientations["dist"].shape == (1, 2, 2)
    assert orientations["omega"].shape == (1, 2, 2)
    assert orientations["theta"].shape == (1, 2, 2)
    assert orientations["phi"].shape == (1, 2, 2)
    # Diagonal distance should be 0
    assert np.isclose(orientations["dist"][0, 0, 0], 0.0)


def test_compute_6d_orientations_fallback_and_gly() -> None:
    """Test with non-standard shape and GLY (missing CB)."""
    # 2 residues: ALA (N, CA, C, O, CB) and GLY (N, CA, C, O)
    # Total atoms = 5 + 4 = 9
    coords = np.random.rand(1, 9, 3)
    atom_names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    residue_indices = [1, 1, 1, 1, 1, 2, 2, 2, 2]

    orientations = compute_6d_orientations(coords, atom_names, residue_indices, 2)

    assert orientations["dist"].shape == (1, 2, 2)
    # Residue 2 (GLY) should have a reconstructed CB
    # We can check that the distance between Res 1 and Res 2 is non-zero
    assert orientations["dist"][0, 0, 1] > 0
    assert not np.isnan(orientations["omega"]).any()
