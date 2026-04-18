"""
Scientific validation of geometry utilities via structural invariants.
Verifies that generated structures satisfy fundamental physical and chemical rules.
"""

import io

import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.geometry.dihedral import calculate_angle, calculate_dihedral
from synth_pdb.geometry.nerf import position_atom_3d_from_internal_coords


def get_peptide_structure(sequence="ACDEFGHIKLMNPQRSTVWY", conformation="alpha", length=None, seed=42):
    """Helper to generate a standard peptide AtomArray."""
    if length:
        sequence = "A" * length
    pdb_content = generate_pdb_content(sequence_str=sequence, conformation=conformation, seed=seed)
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    return pdb_file.get_structure(model=1)

def test_chirality_invariant():
    """
    Invariant: Natural proteins exclusively use L-amino acids.
    Validation: Improper dihedral N-CA-C-CB must be negative (IUPAC).
    """
    peptide = get_peptide_structure()

    # Check all residues with a CB atom (non-Glycine)
    for res_id in np.unique(peptide.res_id):
        res_atoms = peptide[peptide.res_id == res_id]
        res_name = res_atoms.res_name[0]

        if res_name == "GLY":
            continue

        try:
            n = res_atoms[res_atoms.atom_name == "N"].coord[0]
            ca = res_atoms[res_atoms.atom_name == "CA"].coord[0]
            c = res_atoms[res_atoms.atom_name == "C"].coord[0]
            cb = res_atoms[res_atoms.atom_name == "CB"].coord[0]
        except IndexError:
            continue # Missing atoms

        # Improper dihedral N-CA-C-CB
        improper = calculate_dihedral(n, ca, c, cb)

        # Standard L-amino acid improper N-CA-C-CB is ~ -120 deg
        assert -150.0 < improper < -90.0, f"Residue {res_id} {res_name} has non-standard chirality: {improper:.1f}"

def test_peptide_bond_planarity():
    """
    Invariant: Peptide bonds (C-N) are planar due to partial double-bond character.
    Validation: Omega (w) dihedral CA(i)-C(i)-N(i+1)-CA(i+1) should be ~180 (trans) or ~0 (cis).
    """
    peptide = get_peptide_structure(sequence="ACDEFGHIKLMNPQRSTVWY")
    res_ids = np.unique(peptide.res_id)

    for i in range(len(res_ids) - 1):
        r1_id = res_ids[i]
        r2_id = res_ids[i+1]

        r1_atoms = peptide[peptide.res_id == r1_id]
        r2_atoms = peptide[peptide.res_id == r2_id]

        try:
            ca1 = r1_atoms[r1_atoms.atom_name == "CA"].coord[0]
            c1 = r1_atoms[r1_atoms.atom_name == "C"].coord[0]
            n2 = r2_atoms[r2_atoms.atom_name == "N"].coord[0]
            ca2 = r2_atoms[r2_atoms.atom_name == "CA"].coord[0]
        except IndexError:
            continue

        omega = calculate_dihedral(ca1, c1, n2, ca2)

        # Allow for minor deviations (e.g. from minimization or numerical precision)
        # 15 degrees is a safe tolerance for "essentially planar" in relaxed models.
        # Peptide bonds can be either trans (~180) or cis (~0).
        is_trans = abs(abs(omega) - 180.0) < 15.0
        is_cis = abs(omega) < 15.0

        assert is_trans or is_cis, f"Peptide bond {r1_id}-{r2_id} is non-planar: {omega:.1f}"

def test_nerf_cyclic_consistency():
    """
    Invariant: Cartesian -> Internal -> Cartesian transformation should be lossless.
    Validation: Back-calculated coordinates should have near-zero RMSD to original.
    """
    # 1. Start with a known structure
    peptide = get_peptide_structure(length=10)
    orig_coords = peptide.coord[:4].copy() # Use first 4 atoms

    p1, p2, p3, p4_orig = orig_coords[0], orig_coords[1], orig_coords[2], orig_coords[3]

    # 2. Extract internal coordinates
    bond_len = np.linalg.norm(p4_orig - p3)
    bond_ang = calculate_angle(p2, p3, p4_orig)
    dihedral = calculate_dihedral(p1, p2, p3, p4_orig)

    # 3. Reconstruct via NeRF
    p4_recon = position_atom_3d_from_internal_coords(p1, p2, p3, bond_len, bond_ang, dihedral)

    # 4. Verify match
    dist = np.linalg.norm(p4_orig - p4_recon)
    # 1e-7 accounts for minor numerical drift in the full stack
    assert dist < 1e-7, f"NeRF reconstruction failed consistency check. Dist: {dist:.2e}"

def test_ideal_bond_lengths():
    """
    Invariant: Protein backbone bond lengths are highly constrained.
    Validation: Match standard Engh & Huber values within tolerance.
    """
    # Standard values from synth_pdb.data (based on Engh & Huber)
    from synth_pdb.data import BOND_LENGTH_CA_C, BOND_LENGTH_N_CA

    peptide = get_peptide_structure(sequence="AAAAA")

    for res_id in np.unique(peptide.res_id):
        res_atoms = peptide[peptide.res_id == res_id]

        # N-CA
        try:
            n = res_atoms[res_atoms.atom_name == "N"].coord[0]
            ca = res_atoms[res_atoms.atom_name == "CA"].coord[0]
            dist = np.linalg.norm(n - ca)
            # Minimization may shift values slightly from 'ideal'
            assert pytest.approx(dist, abs=0.03) == BOND_LENGTH_N_CA
        except IndexError: pass

        # CA-C
        try:
            ca = res_atoms[res_atoms.atom_name == "CA"].coord[0]
            c = res_atoms[res_atoms.atom_name == "C"].coord[0]
            dist = np.linalg.norm(ca - c)
            assert pytest.approx(dist, abs=0.03) == BOND_LENGTH_CA_C
        except IndexError: pass
