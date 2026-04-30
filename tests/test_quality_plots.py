import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from synth_pdb.quality.plots import (
    apply_publication_style,
    save_publication_figure,
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
)


@pytest.fixture
def mock_matplotlib():
    with patch("synth_pdb.quality.plots.HAS_MATPLOTLIB", True), patch(
        "matplotlib.pyplot.subplots"
    ) as mock_subplots:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        yield mock_fig, mock_ax


def test_apply_publication_style():
    """Verify that publication style is applied without error."""
    try:
        import matplotlib.pyplot as plt

        apply_publication_style()
    except ImportError:
        pytest.skip("matplotlib not installed")


def test_save_publication_figure(tmp_path):
    """Test saving logic with various extensions."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        path = str(tmp_path / "test_fig.png")
        save_publication_figure(fig, path)
        assert os.path.exists(path)

        # Test default extension
        path_no_ext = str(tmp_path / "test_no_ext")
        save_publication_figure(fig, path_no_ext)
        assert os.path.exists(path_no_ext + ".pdf")
    except ImportError:
        pytest.skip("matplotlib not installed")


def test_plot_chemical_shift_correlation(tmp_path):
    """Test chemical shift correlation plot with mock data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    exp_data = {1: {"CA": 55.0, "CB": 20.0}, 2: {"CA": 56.0}}
    syn_data = {1: {"CA": 55.5, "CB": 21.0}, 2: {"CA": 56.5}}

    output_path = str(tmp_path / "corr.png")
    fig = plot_chemical_shift_correlation(
        exp_data, syn_data, atom_type="CA", output_path=output_path
    )

    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_chemical_shift_correlation_insufficient_data():
    """Verify handling of insufficient data for correlation."""
    exp_data = {1: {"CA": 55.0}}
    syn_data = {1: {"CA": 55.5}}

    fig = plot_chemical_shift_correlation(exp_data, syn_data, atom_type="CA")
    assert fig is None


def test_plot_ramachandran_publication(tmp_path):
    """Test Ramachandran publication plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    phi = np.array([-60, -120, -60])
    psi = np.array([-45, 135, -45])

    output_path = str(tmp_path / "rama.png")
    fig = plot_ramachandran_publication(phi, psi, output_path=output_path)

    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_publication(tmp_path):
    """Test SAXS publication plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0, 0.5, 20)
    intensity = np.exp(-(q**2) * 10)

    output_path = str(tmp_path / "saxs.png")
    fig = plot_saxs_publication(q, intensity, rg=15.0, output_path=output_path)

    assert fig is not None
    assert os.path.exists(output_path)


def test_plots_no_matplotlib():
    """Verify that functions return None gracefully when matplotlib is missing."""
    with patch("synth_pdb.quality.plots.HAS_MATPLOTLIB", False):
        assert plot_ramachandran_publication(np.array([]), np.array([])) is None
        assert plot_saxs_publication(np.array([]), np.array([])) is None
        assert plot_chemical_shift_correlation({}, {}) is None


def test_plot_correlation_no_scipy():
    """Verify that correlation plots handle missing scipy gracefully."""
    with patch("synth_pdb.quality.plots.HAS_SCIPY", False), patch(
        "synth_pdb.quality.plots.HAS_MATPLOTLIB", True
    ):
        exp_data = {1: {"CA": 55.0}, 2: {"CA": 56.0}}
        syn_data = {1: {"CA": 55.5}, 2: {"CA": 56.5}}
        fig = plot_chemical_shift_correlation(exp_data, syn_data, atom_type="CA")
        assert fig is None
