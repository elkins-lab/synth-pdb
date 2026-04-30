import numpy as np
import pytest

from synth_pdb.chemical_shifts import calculate_shift_metrics


def test_calculate_shift_metrics_perfect_correlation() -> None:
    """Verify metrics for perfectly identical datasets."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    calc = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["rmsd"] == pytest.approx(0.0)
    assert metrics["correlation"] == pytest.approx(1.0)


def test_calculate_shift_metrics_offset() -> None:
    """Verify RMSD increases with offset but correlation remains 1.0."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    calc = obs + 1.0  # Constant offset

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["rmsd"] == pytest.approx(1.0)
    assert metrics["correlation"] == pytest.approx(1.0)


def test_calculate_shift_metrics_anti_correlated() -> None:
    """Verify correlation is -1.0 for inverse trends."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    calc = np.array([4.0, 3.0, 2.0, 1.0])

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["correlation"] == pytest.approx(-1.0)


def test_calculate_shift_metrics_mismatch_length() -> None:
    """Verify ValueError on array length mismatch."""
    with pytest.raises(ValueError, match="same length"):
        calculate_shift_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_calculate_shift_metrics_empty() -> None:
    """Verify handling of empty arrays."""
    metrics = calculate_shift_metrics(np.array([]), np.array([]))
    assert metrics["rmsd"] == pytest.approx(0.0)
    assert metrics["correlation"] == pytest.approx(0.0)


def test_calculate_shift_metrics_zero_variance() -> None:
    """Verify correlation is 0.0 when one array has zero variance (constant values)."""
    obs = np.array([1.0, 2.0, 3.0])
    calc = np.array([5.0, 5.0, 5.0])  # Constant

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["correlation"] == pytest.approx(0.0)
    # RMSD should still be calculated
    expected_rmsd = np.sqrt(np.mean((obs - calc) ** 2))
    assert np.allclose(metrics["rmsd"], expected_rmsd)


def test_calculate_shift_metrics_single_point() -> None:
    """Verify correlation is 0.0 for single point datasets."""
    metrics = calculate_shift_metrics(np.array([1.0]), np.array([1.1]))
    assert metrics["correlation"] == pytest.approx(0.0)
    assert np.allclose(metrics["rmsd"], 0.1)
