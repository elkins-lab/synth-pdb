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
