"""
Tests for sidechain reconstruction.
"""

import io
from typing import Any, cast

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.geometry.sidechain import reconstruct_sidechain


def test_reconstruct_sidechain_basic() -> None:
    """Test sidechain reconstruction for a simple residue (CYS)."""
    pdb_content = generate_pdb_content(sequence_str="AC", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)

    res_id = 2
    target_chi1 = 60.0
    rotamer: dict[str, Any] = {"chi1": [target_chi1]}

    orig_coords = peptide.coord[peptide.res_id == res_id].copy()
    reconstruct_sidechain(peptide, res_id, rotamer)

    new_coords = peptide.coord[peptide.res_id == res_id]
    assert not np.array_equal(orig_coords, new_coords)

    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    sg_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "SG")].coord[0]

    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, sg_coord))
    if calc_chi1 < 0:
        calc_chi1 += 360
    if target_chi1 < 0:
        target_chi1 += 360

    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1


def test_reconstruct_sidechain_branched_val() -> None:
    """Test sidechain reconstruction for branched residue (VAL)."""
    pdb_content = generate_pdb_content(sequence_str="AV", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)

    res_id = 2
    target_chi1 = 180.0
    rotamer: dict[str, Any] = {"chi1": [target_chi1]}

    reconstruct_sidechain(peptide, res_id, rotamer)

    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    cg1_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CG1")].coord[0]

    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, cg1_coord))
    if calc_chi1 < 0:
        calc_chi1 += 360
    if target_chi1 < 0:
        target_chi1 += 360

    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1


def test_reconstruct_sidechain_invalid_residue() -> None:
    """Test error handling for non-existent residue."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)

    with pytest.raises(ValueError):
        reconstruct_sidechain(peptide, 999, cast(dict[str, Any], {"chi1": [60.0]}))


def test_reconstruct_sidechain_missing_backbone() -> None:
    """Test graceful failure when backbone atoms are missing."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)
    peptide = peptide[peptide.atom_name != "CA"]

    # Should log warning and return early
    reconstruct_sidechain(peptide, 1, cast(dict[str, Any], {"chi1": [60.0]}))


def test_reconstruct_sidechain_unknown_residue() -> None:
    """Test handling of unknown residue type (hitting Miss 83-85)."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)

    # Change ALL residue names to something unknown
    peptide.res_name[:] = "INVALID123"

    # Should log warning and return
    reconstruct_sidechain(peptide, 1, cast(dict[str, Any], {"chi1": [60.0]}))


def test_reconstruct_sidechain_missing_template_atoms(mocker: Any) -> None:
    """Test handling of templates missing essential backbone atoms (hitting Miss 96-97)."""
    # Use VAL for mock to avoid affecting ALA setup in generate_pdb_content
    bad_template = struc.AtomArray(1)
    bad_template[0] = struc.Atom(res_id=1, res_name="VAL", atom_name="N", coord=[0, 0, 0])

    pdb_content = generate_pdb_content(sequence_str="V", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)

    mocker.patch("biotite.structure.info.residue", return_value=bad_template)
    reconstruct_sidechain(peptide, 1, cast(dict[str, Any], {"chi1": [60.0]}))


def test_rotate_points() -> None:
    """Test the low-level rotate_points JIT function."""
    from synth_pdb.geometry.sidechain import rotate_points

    # Points on the Y axis
    points = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    # Axis is the Z axis (0,0,0 to 0,0,1)
    axis_p1 = np.array([0.0, 0.0, 0.0])
    axis_p2 = np.array([0.0, 0.0, 1.0])

    # Rotate 90 degrees around Z: Y becomes -X
    rotated = rotate_points(points, axis_p1, axis_p2, 90.0)

    expected = np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    assert np.allclose(rotated, expected, atol=1e-5)


def test_reconstruct_sidechain_no_chi1() -> None:
    """Test path where chi1 is missing from rotamer dict."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)
    # Should return early
    reconstruct_sidechain(peptide, 1, cast(dict[str, Any], {"prob": 0.5}))


def test_reconstruct_sidechain_missing_gamma() -> None:
    """Test path where gamma atom is missing in template (e.g. ALA)."""
    # ALA doesn't have gamma, so chi1 doesn't apply
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(cast(str, pdb_content)))
    peptide = pdb_file.get_structure(model=1)
    # Reconstruct chi1 on ALA (which has no CG/OG/SG)
    reconstruct_sidechain(peptide, 1, cast(dict[str, Any], {"chi1": 60.0}))
    # Should not crash and should complete
