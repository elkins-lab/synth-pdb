import numpy as np
import pytest

from synth_pdb.chemical_shifts import calculate_shift_metrics


def test_shift_metrics_perfect_fit() -> None:
    """If observed and calculated shifts are identical, metrics should be perfect."""
    obs = np.array([4.5, 4.8, 5.2])
    calc = np.array([4.5, 4.8, 5.2])

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["rmsd"] == 0.0
    assert metrics["correlation"] == 1.0


def test_shift_metrics_standard() -> None:
    """Test metrics with a known set of values."""
    # obs = [5.0, 6.0], calc = [5.1, 5.9]
    # mean_obs = 5.5, mean_calc = 5.5
    # diff = [-0.1, 0.1] -> sum(diff^2) = 0.02 -> mean = 0.01 -> sqrt = 0.1 (RMSD)
    # correlation: should be -1.0 since trends are opposite but let's use positive trends
    obs = np.array([5.0, 6.0])
    calc = np.array([5.1, 6.1])  # Identical trend, offset by 0.1

    metrics = calculate_shift_metrics(obs, calc)
    assert metrics["rmsd"] == pytest.approx(0.1)
    assert metrics["correlation"] == pytest.approx(1.0)


def test_shift_metrics_empty() -> None:
    """Empty arrays should handle gracefully."""
    metrics = calculate_shift_metrics(np.array([]), np.array([]))
    assert metrics["rmsd"] == 0.0
    assert metrics["correlation"] == 0.0
