"""
Tests to fill coverage gaps in the geometry module.
"""

import biotite.structure as struc
import numpy as np

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
    p3 = np.array([[2.0, 0.0, 0.0]]) # Collinear

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
    peptide += struc.array([
        struc.Atom(res_id=1, res_name="ALA", atom_name="N", coord=[-1, 0, 0], chain_id="A"),
        struc.Atom(res_id=1, res_name="ALA", atom_name="C", coord=[0, 1, 0], chain_id="A")
    ])

    orig_coords = peptide.coord.copy()
    reconstruct_sidechain(peptide, 1, {"chi2": [180.0]}) # No chi1
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

def test_calculate_rmsd_empty():
    """Test calculate_rmsd with empty or zero-size arrays."""
    from synth_pdb.geometry.rmsd import calculate_rmsd

    P = np.array([]).reshape(0, 3)
    Q = np.array([]).reshape(0, 3)
    assert calculate_rmsd(P, Q) == 0.0

    P = np.zeros((1, 3))
    Q = np.zeros((1, 3))
    assert calculate_rmsd(P, Q) == 0.0
