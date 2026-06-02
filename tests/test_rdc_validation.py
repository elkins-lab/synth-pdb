import numpy as np
import pytest

from synth_pdb.rdc import calculate_rdc_q_factor


def test_rdc_q_factor_perfect_fit() -> None:
    """If observed and calculated RDCs are identical, Q-factor should be 0.0."""
    obs = np.array([10.5, -5.2, 18.0])
    calc = np.array([10.5, -5.2, 18.0])

    q = calculate_rdc_q_factor(obs, calc)
    assert q == 0.0


def test_rdc_q_factor_standard() -> None:
    """Test Q-factor with a known set of values.
    Q = sqrt(sum(obs-calc)^2) / sqrt(sum(obs^2))
    obs = [10, 20]
    calc = [11, 19]
    diff = [-1, 1] -> sum(diff^2) = 2 -> sqrt(2) = 1.414
    obs^2 = [100, 400] -> sum(obs^2) = 500 -> sqrt(500) = 22.36
    Q = 1.414 / 22.36 = 0.0632
    """
    obs = np.array([10.0, 20.0])
    calc = np.array([11.0, 19.0])

    q = calculate_rdc_q_factor(obs, calc)
    assert q == pytest.approx(0.063245, abs=0.0001)


def test_rdc_q_factor_empty() -> None:
    """Empty arrays should return nan (undefined)."""
    q = calculate_rdc_q_factor(np.array([]), np.array([]))
    assert np.isnan(q)
