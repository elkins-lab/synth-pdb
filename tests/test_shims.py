import pytest
from synth_pdb import nef_io
from synth_pdb import structure_utils


def test_nef_io_exports():
    """Verify that nef_io re-exports expected functions from synth-nmr."""
    assert hasattr(nef_io, "read_nef_restraints")
    assert hasattr(nef_io, "write_nef_file")
    assert hasattr(nef_io, "write_nef_relaxation")
    assert hasattr(nef_io, "write_nef_chemical_shifts")

    # Check __all__
    for name in nef_io.__all__:
        assert hasattr(nef_io, name)


def test_structure_utils_exports():
    """Verify that structure_utils re-exports expected functions from synth-nmr."""
    assert hasattr(structure_utils, "get_secondary_structure")

    # Check __all__
    for name in structure_utils.__all__:
        assert hasattr(structure_utils, name)


def test_nef_io_functional_smoke():
    """Smoke test for nef_io to ensure it doesn't crash on import/access."""
    try:
        import synth_nmr.nef_io as _nef

        assert nef_io.read_nef_restraints == _nef.read_nef_restraints
    except ImportError:
        pytest.skip("synth-nmr not installed")


def test_structure_utils_functional_smoke():
    """Smoke test for structure_utils to ensure it doesn't crash on call."""
    import numpy as np
    import biotite.structure as struc

    # Minimal structure
    atoms = struc.AtomArray(5)
    atoms.res_id = np.array([1, 1, 1, 1, 1])
    atoms.atom_name = np.array(["N", "CA", "C", "O", "CB"])
    atoms.coord = np.random.rand(5, 3)

    try:
        # This should call the re-exported synth-nmr function
        ss = structure_utils.get_secondary_structure(atoms)
        assert isinstance(ss, list | np.ndarray)
    except ImportError:
        pytest.skip("synth-nmr not installed")


def test_nef_io_read_functional():
    """Verify that nef_io.read_nef_restraints is functional."""
    import os
    import tempfile

    nef_content = """save_test
_nef_distance_restraint_list.sf_category nef_distance_restraint_list
loop_
_nef_distance_restraint.index
_nef_distance_restraint.chain_code_1
_nef_distance_restraint.sequence_code_1
_nef_distance_restraint.residue_name_1
_nef_distance_restraint.atom_name_1
_nef_distance_restraint.chain_code_2
_nef_distance_restraint.sequence_code_2
_nef_distance_restraint.residue_name_2
_nef_distance_restraint.atom_name_2
_nef_distance_restraint.target_value
1 A 1 ALA N A 2 GLY CA 5.0
stop_
save_
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nef", delete=False) as tf:
        tf.write(nef_content)
        temp_path = tf.name

    try:
        restraints = nef_io.read_nef_restraints(temp_path)
        assert len(restraints) == 1
        assert restraints[0]["res_1"] == "ALA"
    except ImportError:
        pytest.skip("synth-nmr not installed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
