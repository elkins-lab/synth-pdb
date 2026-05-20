import numpy as np
import biotite.structure as struc
import pytest
from unittest.mock import patch
from synth_pdb.saxs import HAS_SYNTH_SAXS, calculate_saxs_profile, calculate_radius_of_gyration


@pytest.mark.skipif(not HAS_SYNTH_SAXS, reason="synth-saxs not installed")
def test_saxs_shim_integration():
    """Verify that the synth-pdb shim correctly re-exports synth-saxs."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0, 0, 0]])
    atoms.element = ["C"]

    q, I = calculate_saxs_profile(atoms, n_points=5)
    assert len(q) == 5
    assert len(I) == 5
    assert I[0] > 0


@pytest.mark.skipif(not HAS_SYNTH_SAXS, reason="synth-saxs not installed")
def test_rg_shim():
    """Verify Rg calculation remains available via shim."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0, 0, 0]])
    atoms.element = ["C"]
    atoms.res_name = ["ALA"]
    rg = calculate_radius_of_gyration(atoms)
    assert rg == 0.0


def test_visualization_saxs_missing_deps():
    """Verify that plot_saxs_results fallback raises ImportError."""
    import importlib
    import synth_pdb.visualization_saxs

    # 1. Simulate missing synth_saxs and reload module to trigger fallback definition
    with patch.dict("sys.modules", {"synth_saxs": None}):
        importlib.reload(synth_pdb.visualization_saxs)
        from synth_pdb.visualization_saxs import plot_saxs_results

        with pytest.raises(ImportError, match="plot_saxs_results requires synth-saxs"):
            plot_saxs_results()

    # 2. Cleanup: Restore the module to its original state (with real synth_saxs if present)
    importlib.reload(synth_pdb.visualization_saxs)
