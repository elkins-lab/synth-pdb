import numpy as np
import biotite.structure as struc
import pytest
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
