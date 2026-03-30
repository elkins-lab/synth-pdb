"""
Tests for sidechain reconstruction.
"""

import io

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.geometry.sidechain import reconstruct_sidechain


def test_reconstruct_sidechain_basic():
    """Test sidechain reconstruction for a simple residue (CYS)."""
    pdb_content = generate_pdb_content(sequence_str="AC", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)

    res_id = 2
    target_chi1 = 60.0
    rotamer = {"chi1": [target_chi1]}

    orig_coords = peptide.coord[peptide.res_id == res_id].copy()
    reconstruct_sidechain(peptide, res_id, rotamer)

    new_coords = peptide.coord[peptide.res_id == res_id]
    assert not np.array_equal(orig_coords, new_coords)

    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    sg_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "SG")].coord[0]

    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, sg_coord))
    if calc_chi1 < 0: calc_chi1 += 360
    if target_chi1 < 0: target_chi1 += 360

    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1

def test_reconstruct_sidechain_branched_val():
    """Test sidechain reconstruction for branched residue (VAL)."""
    pdb_content = generate_pdb_content(sequence_str="AV", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)

    res_id = 2
    target_chi1 = 180.0
    rotamer = {"chi1": [target_chi1]}

    reconstruct_sidechain(peptide, res_id, rotamer)

    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    cg1_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CG1")].coord[0]

    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, cg1_coord))
    if calc_chi1 < 0: calc_chi1 += 360
    if target_chi1 < 0: target_chi1 += 360

    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1

def test_reconstruct_sidechain_invalid_residue():
    """Test error handling for non-existent residue."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)

    with pytest.raises(ValueError):
        reconstruct_sidechain(peptide, 999, {"chi1": [60.0]})

def test_reconstruct_sidechain_missing_backbone():
    """Test graceful failure when backbone atoms are missing."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    peptide = peptide[peptide.atom_name != "CA"]

    # Should log warning and return None
    result = reconstruct_sidechain(peptide, 1, {"chi1": [60.0]})
    assert result is None
