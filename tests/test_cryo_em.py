import os
from typing import Any

import biotite.structure as struc
import numpy as np
import pytest

from synth_pdb.cryo_em import CryoEMSimulator, generate_density_map, save_mrc_file


def test_generate_density_map_shape() -> None:
    """Verify that the generated density map has the correct shape and origin."""
    # Create a simple structure (2 atoms)
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [10, 10, 10]])
    atoms.element = ["C", "C"]

    # Generate map with 1.0A spacing and 5.0A buffer
    # Expected box: [-5, -5, -5] to [15, 15, 15] -> 20x20x20
    density, origin = generate_density_map(atoms, grid_spacing=1.0, buffer=5.0)

    assert density.shape == (20, 20, 20)
    assert np.allclose(origin, np.array([-5.0, -5.0, -5.0]))
    assert np.max(density) > 0


def test_generate_density_from_stack() -> None:
    """Verify density generation from an ensemble (AtomArrayStack)."""
    stack = struc.AtomArrayStack(2, 1)
    stack.coord[0, 0] = [0, 0, 0]
    stack.coord[1, 0] = [2, 2, 2]
    stack.element = ["C"]

    # 2 models, 1 atom each. Resolution 3A.
    density, origin = generate_density_map(stack, resolution=3.0, grid_spacing=1.0, buffer=5.0)

    # The density should be "smeared" between the two positions
    assert np.max(density) > 0
    # Center of mass is roughly [1, 1, 1]
    # Origin is roughly [-5, -5, -5]
    # Center index is roughly [6, 6, 6]
    assert density[6, 6, 6] > 0


def test_save_mrc_file(tmp_path: Any) -> None:
    """Verify that MRC files can be saved and re-read."""
    import mrcfile

    path = str(tmp_path / "test.mrc")
    density = np.random.rand(10, 10, 10).astype(np.float32)
    origin = np.array([1.2, 3.4, 5.6])

    save_mrc_file(path, density, origin, spacing=1.0)

    assert os.path.exists(path)

    with mrcfile.open(path) as mrc:
        assert mrc.data.shape == (10, 10, 10)
        assert np.allclose(mrc.data, density)
        assert mrc.header.origin.x == pytest.approx(1.2)
        assert mrc.voxel_size.x == pytest.approx(1.0)


def test_cryo_em_simulator_wrapper() -> None:
    """Test the CryoEMSimulator class wrapper."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0, 0, 0]])
    atoms.element = ["C"]

    sim = CryoEMSimulator(resolution=5.0, spacing=2.0)
    density = sim.simulate(atoms)

    # With 2A spacing and 5A buffer, box is roughly [-5, 5] -> 10A -> 5 voxels
    assert len(density.shape) == 3
    assert np.max(density) > 0
