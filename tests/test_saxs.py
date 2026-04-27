import os
from typing import Any

import biotite.structure as struc
import numpy as np
import pytest

from synth_pdb.saxs import SaxsSimulator, calculate_saxs_profile, export_saxs_profile


def test_saxs_profile_shape() -> None:
    """Verify that the generated SAXS profile has the correct length and basic decay in vacuum."""
    # Create a simple structure (2 atoms)
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [10, 10, 10]])
    atoms.element = ["C", "C"]

    n_points = 21
    # Disable solvent for monotonic shape check
    q, intensity = calculate_saxs_profile(atoms, n_points=n_points, include_solvent=False)

    assert len(q) == n_points
    assert len(intensity) == n_points
    assert intensity[0] > intensity[-1]  # Decay check in vacuum


def test_saxs_guinier_region() -> None:
    """Verify that the I(q) curve is well-behaved at low q."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0, 0, 0]])
    atoms.element = ["C"]

    q, intensity = calculate_saxs_profile(
        atoms, q_min=0.0, q_max=0.01, n_points=5, include_solvent=False
    )

    # For a single atom in vacuum, scattering should be positive and nearly flat at extremely low q
    assert intensity[0] > 0
    assert np.abs(intensity[0] - intensity[1]) < 0.01


def test_saxs_simulator_ensemble() -> None:
    """Verify ensemble averaging in SAXS simulation."""
    stack = struc.AtomArrayStack(2, 1)
    stack.coord[0, 0] = [0, 0, 0]
    stack.coord[1, 0] = [5, 5, 5]
    stack.element = ["C"]

    sim = SaxsSimulator(n_points=10)
    intensity = sim.simulate(stack)

    assert len(intensity) == 10
    assert np.all(intensity >= 0)


def test_saxs_simulator_single_structure() -> None:
    """Verify SAXS simulator works with a single AtomArray."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0, 0, 0]])
    atoms.element = ["C"]

    sim = SaxsSimulator(n_points=5)
    intensity = sim.simulate(atoms)

    assert len(intensity) == 5
    assert np.all(intensity > 0)


def test_get_form_factor_fallback() -> None:
    """Verify that get_form_factor falls back to Carbon for unknown elements."""
    from synth_pdb.saxs import get_form_factor

    q = np.array([0.1])
    f_carbon = get_form_factor("C", q)
    f_unknown = get_form_factor("UnknownElement", q)

    assert np.allclose(f_carbon, f_unknown)


def test_export_saxs(tmp_path: Any) -> None:
    """Verify SAXS data export."""
    path = str(tmp_path / "test.dat")
    q = np.linspace(0, 0.5, 10)
    intensity = np.random.rand(10)

    export_saxs_profile(q, intensity, path)

    assert os.path.exists(path)
    # Check if we can read it back
    data = np.loadtxt(path)
    assert data.shape == (10, 3)
    assert np.allclose(data[:, 0], q)


def test_saxs_visualization(tmp_path: Any) -> None:
    """Verify that SAXS plots can be generated."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from synth_pdb.visualization_saxs import plot_saxs_results

    q = np.linspace(0, 0.5, 50)
    intensity = np.exp(-(q**2) * 10)

    output_path = str(tmp_path / "saxs_plot.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="all", rg=15.0)

    assert fig is not None
    assert os.path.exists(output_path)
