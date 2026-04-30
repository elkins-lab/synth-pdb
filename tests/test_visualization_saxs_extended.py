import os
import pytest
import numpy as np
from synth_pdb.visualization_saxs import plot_saxs_results


def test_plot_saxs_results_standard(tmp_path):
    """Test standard SAXS plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    output_path = str(tmp_path / "saxs_std.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="standard")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_kratky(tmp_path):
    """Test Kratky plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    output_path = str(tmp_path / "saxs_kratky.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="kratky")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_guinier(tmp_path):
    """Test Guinier plot and Rg estimation logic."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.001, 0.1, 50)
    # I(q) = I(0) * exp(-q^2 * Rg^2 / 3)
    rg_target = 20.0
    intensity = 100 * np.exp(-(q**2) * (rg_target**2) / 3.0)

    output_path = str(tmp_path / "saxs_guinier.png")
    fig = plot_saxs_results(
        q, intensity, output_path=output_path, plot_type="guinier", rg=rg_target
    )
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_no_matplotlib():
    """Verify graceful failure when matplotlib is missing."""
    with pytest.MonkeyPatch().context() as m:
        import synth_pdb.visualization_saxs

        m.setattr(synth_pdb.visualization_saxs, "HAS_MATPLOTLIB", False)
        fig = plot_saxs_results(np.array([0.1]), np.array([1.0]))
        assert fig is None
