import os
import tempfile
import numpy as np
import biotite.structure as struc
from synth_pdb.distogram import calculate_distogram, export_distogram


def test_calculate_distogram_cb_fallback() -> None:
    """Test CB method with fallback (GLY)."""
    # Create 1 res GLY
    atoms = struc.AtomArray(1)
    atoms.res_name = np.array(["GLY"])
    atoms.atom_name = np.array(["CA"])
    atoms.coord = np.array([[0, 0, 0]])

    # Should fallback to CA
    dist = calculate_distogram(atoms, method="cb")
    assert dist.shape == (1, 1)


def test_export_distogram_variants() -> None:
    """Test all export formats."""
    matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
    with tempfile.TemporaryDirectory() as tmp:
        # JSON
        json_path = os.path.join(tmp, "d.json")
        export_distogram(matrix, json_path, fmt="json")
        assert os.path.exists(json_path)

        # CSV
        csv_path = os.path.join(tmp, "d.csv")
        export_distogram(matrix, csv_path, fmt="csv")
        assert os.path.exists(csv_path)

        # NPZ
        npz_path = os.path.join(tmp, "d.npz")
        export_distogram(matrix, npz_path, fmt="npz")
        assert os.path.exists(npz_path)


import pytest


def test_export_distogram_invalid() -> None:
    """Test invalid format raises error."""
    matrix = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Unknown format"):
        export_distogram(matrix, "test.txt", fmt="invalid")
