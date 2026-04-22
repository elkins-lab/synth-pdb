import numpy as np
import pytest

from synth_pdb.geometry.rmsd import (
    calculate_average_coords,
    calculate_rmsd,
    calculate_rmsd_to_average,
)
from synth_pdb.geometry.superposition import apply_transformation, kabsch_superposition


def test_rmsd_identical_structures() -> None:
    """RMSD of identical structures should be 0.0."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert calculate_rmsd(coords, coords) == 0.0


def test_rmsd_mismatch_shape() -> None:
    """RMSD should raise ValueError on shape mismatch."""
    c1 = np.zeros((3, 3))
    c2 = np.zeros((4, 3))
    with pytest.raises(ValueError, match="same shape"):
        calculate_rmsd(c1, c2)


def test_rmsd_zero_points() -> None:
    """RMSD of zero points should return 0.0 and log warning (internally)."""
    c1 = np.zeros((0, 3))
    c2 = np.zeros((0, 3))
    assert calculate_rmsd(c1, c2) == 0.0


def test_kabsch_identical() -> None:
    """Kabsch on identical structures should return identity rotation and zero translation."""
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation, translation = kabsch_superposition(coords, coords)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, np.zeros(3))


def test_kabsch_translation_only() -> None:
    """Kabsch should correctly identify pure translation."""
    c1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offset = np.array([10.0, 20.0, 30.0])
    c2 = c1 + offset

    rotation, translation = kabsch_superposition(c1, c2)
    assert np.allclose(rotation, np.eye(3), atol=1e-7)
    assert np.allclose(translation, offset, atol=1e-7)


def test_kabsch_degenerate_line() -> None:
    """Kabsch should handle cases where all points are collinear."""
    # Points on X axis
    c1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Rotated 90 deg around Z
    c2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])

    rotation, translation = kabsch_superposition(c1, c2)
    # Even if degenerate, it should return a valid finite rotation
    assert np.all(np.isfinite(rotation))
    assert np.all(np.isfinite(translation))

    # Check if superposition actually reduces RMSD to near zero
    aligned = apply_transformation(c1, rotation, translation)
    assert calculate_rmsd(aligned, c2) < 1e-7


def test_kabsch_nan_handling() -> None:
    """Kabsch should handle NaNs by returning identity/zero and not crashing."""
    c1 = np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    c2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    rotation, translation = kabsch_superposition(c1, c2)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, np.zeros(3))


def test_calculate_average_coords_empty() -> None:
    """Average coords of empty list should be empty array."""
    assert calculate_average_coords([]).size == 0


def test_calculate_rmsd_to_average_single_structure() -> None:
    """RMSD to average for a single structure should be 0.0."""
    c1 = np.random.rand(5, 3)
    avg_rmsd, avg_coords = calculate_rmsd_to_average([c1])
    assert avg_rmsd == 0.0
    assert np.allclose(avg_coords, c1)


def test_calculate_rmsd_to_average_empty() -> None:
    """RMSD to average for empty list should return NaN."""
    avg_rmsd, avg_coords = calculate_rmsd_to_average([])
    assert np.isnan(avg_rmsd)
    assert avg_coords.size == 0
