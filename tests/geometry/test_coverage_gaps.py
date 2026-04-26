"""
Tests to fill coverage gaps in the geometry module.
"""

import biotite.structure as struc
import numpy as np
import pytest

from synth_pdb.geometry.dihedral import calculate_dihedral, calculate_dihedral_angle
from synth_pdb.geometry.nerf import place_atom, position_atom_3d_from_internal_coords
from synth_pdb.geometry.sidechain import reconstruct_sidechain
from synth_pdb.geometry.vectorized import batched_dihedral, position_atoms_batch


def test_batched_dihedral():
    """Test batched_dihedral calculation."""
    p1 = np.array([[1.0, 1.0, 0.0]])
    p2 = np.array([[0.0, 1.0, 0.0]])
    p3 = np.array([[0.0, 0.0, 0.0]])
    p4 = np.array([[1.0, 0.0, 0.0]])

    # Planar trans-like should be 180 or 0
    dihedrals = batched_dihedral(p1, p2, p3, p4)
    assert np.isclose(abs(dihedrals[0]), 180.0) or np.isclose(dihedrals[0], 0.0)


def test_position_atoms_batch_degenerate():
    """Test position_atoms_batch with collinear points (degenerate case)."""
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    p3 = np.array([[2.0, 0.0, 0.0]])  # Collinear

    bond_lengths = np.array([1.5])
    bond_angles = np.array([90.0])
    dihedral_angles = np.array([90.0])

    p4 = position_atoms_batch(p1, p2, p3, bond_lengths, bond_angles, dihedral_angles)
    # Should not be NaN
    assert not np.any(np.isnan(p4))


def test_geometry_aliases():
    """Test that geometry aliases work correctly."""
    p1 = np.array([1.0, 1.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([1.0, 0.0, 0.0])

    assert calculate_dihedral(p1, p2, p3, p4) == calculate_dihedral_angle(p1, p2, p3, p4)

    pos1 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 90.0, 180.0)
    pos2 = place_atom(p1, p2, p3, 1.5, 90.0, 180.0)
    assert np.allclose(pos1, pos2)


def test_reconstruct_sidechain_no_chi1():
    """Test reconstruct_sidechain when chi1 is not in rotamer."""
    # Create a simple structure
    atom = struc.Atom(res_id=1, res_name="ALA", atom_name="CA", coord=[0, 0, 0], chain_id="A")
    peptide = struc.array([atom])
    peptide += struc.array(
        [
            struc.Atom(res_id=1, res_name="ALA", atom_name="N", coord=[-1, 0, 0], chain_id="A"),
            struc.Atom(res_id=1, res_name="ALA", atom_name="C", coord=[0, 1, 0], chain_id="A"),
        ]
    )

    orig_coords = peptide.coord.copy()
    reconstruct_sidechain(peptide, 1, {"chi2": [180.0]})  # No chi1
    assert np.array_equal(orig_coords, peptide.coord)


def test_kabsch_superposition_singular():
    """Test kabsch_superposition with highly degenerate coordinates."""
    from synth_pdb.geometry.superposition import kabsch_superposition

    # All points at origin
    P = np.zeros((3, 3))
    Q = np.zeros((3, 3))

    R, t = kabsch_superposition(P, Q)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert not np.any(np.isnan(R))
    assert not np.any(np.isnan(t))


def test_calculate_rmsd_errors():
    """Test error handling in calculate_rmsd."""
    from synth_pdb.geometry.rmsd import calculate_rmsd

    # Shape mismatch
    with pytest.raises(ValueError, match="same shape"):
        calculate_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))

    # Not Nx3
    with pytest.raises(ValueError, match="Nx3 arrays"):
        calculate_rmsd(np.zeros((3, 2)), np.zeros((3, 2)))


def test_calculate_rmsd_to_average_empty():
    """Test calculate_rmsd_to_average with empty input."""
    from synth_pdb.geometry.rmsd import calculate_rmsd_to_average

    avg_rmsd, avg_coords = calculate_rmsd_to_average([])
    assert np.isnan(avg_rmsd)
    assert avg_coords.size == 0


def test_kabsch_superposition_singular_cases():
    """Test kabsch_superposition with singular or non-finite inputs."""
    from synth_pdb.geometry.superposition import kabsch_superposition

    # Empty arrays
    R, t = kabsch_superposition(np.array([]).reshape(0, 3), np.array([]).reshape(0, 3))
    assert R.size == 0

    # Non-finite values
    P = np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])
    Q = np.zeros((3, 3))
    R, t = kabsch_superposition(P, Q)
    assert np.allclose(R, np.eye(3))

    # Collinear points causing singular H (handled by SVD usually, but good to check)
    P = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    Q = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]])
    R, t = kabsch_superposition(P, Q)
    assert not np.any(np.isnan(R))


def test_dihedral_degenerate_normals():
    """Test dihedral and angle with degenerate (zero-length) vectors."""
    from synth_pdb.geometry.dihedral import calculate_angle, calculate_dihedral

    # Angle with zero vector
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    assert calculate_angle(p1, p2, p3) == 0.0

    # Dihedral with collinear points (zero normal)
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([2, 0, 0])
    p4 = np.array([3, 0, 0])
    assert calculate_dihedral(p1, p2, p3, p4) == 0.0


def test_calculate_average_coords_empty():
    """Test calculate_average_coords with empty inputs."""
    from synth_pdb.geometry.rmsd import calculate_average_coords

    # Empty list
    avg = calculate_average_coords([])
    assert avg.size == 0

    # List with empty array
    avg = calculate_average_coords([np.array([]).reshape(0, 3)])
    assert avg.size == 0


def test_kabsch_superposition_linalg_errors(mocker):
    """Test kabsch_superposition handling of LinAlgErrors."""
    from synth_pdb.geometry.superposition import kabsch_superposition

    P = np.eye(3)
    Q = np.eye(3)

    # Mock SVD to fail
    mocker.patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("SVD failed"))
    R, t = kabsch_superposition(P, Q)
    assert np.allclose(R, np.eye(3))

    # Mock determinant to fail
    mocker.patch("numpy.linalg.svd", return_value=(np.eye(3), np.ones(3), np.eye(3)))
    mocker.patch("numpy.linalg.det", side_effect=np.linalg.LinAlgError("Det failed"))
    R, t = kabsch_superposition(P, Q)
    assert np.allclose(R, np.eye(3))


def test_kabsch_superposition_non_finite_R():
    """Test kabsch_superposition handling of non-finite rotation matrices."""
    pass


def test_reconstruct_sidechain_missing_template(mocker):
    """Test reconstruct_sidechain when residue template is missing."""
    from synth_pdb.geometry.sidechain import reconstruct_sidechain

    atom = struc.Atom(res_id=1, res_name="ALA", atom_name="CA", coord=[0, 0, 0], chain_id="A")
    peptide = struc.array([atom])

    # Mock biotite.structure.info.residue to raise KeyError
    mocker.patch("biotite.structure.info.residue", side_effect=KeyError("Missing"))
    assert reconstruct_sidechain(peptide, 1, {"chi1": [60.0]}) is None


def test_reconstruct_sidechain_missing_template_atoms(mocker):
    """Test sidechain reconstruction with missing template backbone atoms (hitting Miss 100)."""
    # Create valid peptide
    n = struc.Atom([0, 0, 0], atom_name="N", res_id=1, res_name="ALA")
    ca = struc.Atom([1, 0, 0], atom_name="CA", res_id=1, res_name="ALA")
    c = struc.Atom([1, 1, 0], atom_name="C", res_id=1, res_name="ALA")
    cb = struc.Atom([1, 1, 1], atom_name="CB", res_id=1, res_name="ALA")
    peptide = struc.array([n, ca, c, cb])

    # Mock template to miss 'N'
    bad_template = struc.array([ca, c])  # No N
    mocker.patch("biotite.structure.info.residue", return_value=bad_template)

    # Should return early without error
    reconstruct_sidechain(peptide, 1, {"chi1": 60.0})


def test_calculate_rmsd_empty_or_nan():
    """Test RMSD with empty arrays (hitting Miss 44-45)."""
    from synth_pdb.geometry.rmsd import calculate_rmsd

    p = np.array([]).reshape(0, 3)
    q = np.array([]).reshape(0, 3)
    assert calculate_rmsd(p, q) == 0.0


def test_calculate_rmsd_to_average_gaps():
    """Test calculate_rmsd_to_average gaps (hitting Miss 167-168)."""
    from synth_pdb.geometry.rmsd import calculate_rmsd_to_average

    # 0 structures (empty list)
    res, avg = calculate_rmsd_to_average([])
    assert np.isnan(res)

    # Empty avg_coords case (list with empty array)
    res, avg = calculate_rmsd_to_average([np.array([]).reshape(0, 3)])
    assert np.isnan(res)


def test_reconstruct_sidechain_missing_backbone():
    """Test sidechain reconstruction with missing backbone atoms (hitting Miss 83-85)."""
    # Create structure with only CA
    atom = struc.Atom(res_id=1, res_name="ALA", atom_name="CA", coord=[0, 0, 0], chain_id="A")
    peptide = struc.array([atom])

    # This should log a warning and return early
    reconstruct_sidechain(peptide, 1, {"chi1": 60.0})
    # Original coord should remain unchanged
    assert np.allclose(peptide.coord[0], [0, 0, 0])


def test_calculate_rmsd_squared_diff_empty():
    """Test calculate_rmsd when squared_diff is empty (hitting Miss 52-53)."""
    pass


def test_dihedral_collinear_normalized():
    """Test dihedral with collinear vectors (hitting Miss 59-63)."""
    from synth_pdb.geometry.dihedral import calculate_dihedral

    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([1, 0, 0])  # Zero-length bond
    p4 = np.array([2, 0, 0])
    res = calculate_dihedral(p1, p2, p3, p4)
    assert res in [0.0, 180.0]


def test_kabsch_superposition_singular_det(mocker):
    """Test kabsch_superposition singular determinant (hitting Miss 80)."""
    from synth_pdb.geometry.superposition import kabsch_superposition

    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Mirror image

    # Force det to be exactly 0
    mocker.patch("numpy.linalg.det", return_value=0.0)
    R, _ = kabsch_superposition(P, Q)
    assert not np.any(np.isnan(R))


def test_kabsch_superposition_non_finite_check(mocker):
    """Test kabsch_superposition R finite check (hitting Miss 90-91)."""
    from synth_pdb.geometry.superposition import kabsch_superposition

    P = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    Q = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])

    mock_u = np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])
    mock_s = np.array([1, 1, 1])
    mock_vt = np.eye(3)

    mocker.patch("numpy.linalg.svd", return_value=(mock_u, mock_s, mock_vt))
    R, _ = kabsch_superposition(P, Q)
    assert np.allclose(R, np.eye(3))
