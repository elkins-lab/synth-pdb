"""
Unit tests for geometry.rmsd module.

Tests RMSD calculation functions including basic RMSD, pairwise RMSD,
average coordinates, and RMSD to average.
"""

import numpy as np
import pytest

from synth_pdb.geometry.rmsd import (
    calculate_average_coords,
    calculate_pairwise_rmsd,
    calculate_rmsd,
    calculate_rmsd_to_average,
)


class TestCalculateRMSD:
    """Test calculate_rmsd function."""

    def test_identical_structures(self):
        """Should return 0.0 for identical structures."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        assert rmsd == pytest.approx(0.0)

    def test_unit_translation(self):
        """Should calculate RMSD for unit translation."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        # All points translated by [1, 0, 0]
        # RMSD = sqrt(mean([1^2, 1^2, 1^2])) = 1.0
        assert rmsd == pytest.approx(1.0)

    def test_known_rmsd_value(self):
        """Should calculate correct RMSD for known case."""
        # Create structures with known RMSD
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        # Point 1: diff = 0, Point 2: diff = [0, 1, 0]
        # RMSD = sqrt(mean([0, 1])) = sqrt(0.5) ≈ 0.707
        assert rmsd == pytest.approx(np.sqrt(0.5))

    def test_345_triangle(self):
        """Should calculate RMSD for 3-4-5 triangle displacement."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[3.0, 4.0, 0.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        # Distance = sqrt(3^2 + 4^2) = 5
        assert rmsd == pytest.approx(5.0)

    def test_empty_arrays(self):
        """Should return 0.0 for empty coordinate arrays."""
        coords1 = np.array([]).reshape(0, 3)
        coords2 = np.array([]).reshape(0, 3)

        rmsd = calculate_rmsd(coords1, coords2)

        assert rmsd == pytest.approx(0.0)

    def test_single_atom(self):
        """Should handle single atom RMSD."""
        coords1 = np.array([[1.0, 2.0, 3.0]])
        coords2 = np.array([[1.5, 2.5, 3.5]])

        rmsd = calculate_rmsd(coords1, coords2)

        # Distance = sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75)
        expected = np.sqrt(0.75)
        assert rmsd == pytest.approx(expected)

    def test_multiple_atoms(self):
        """Should calculate RMSD for multiple atoms."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        coords2 = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.1, 0.0, 1.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        # All points shifted by 0.1 in x
        # RMSD = sqrt(mean([0.01, 0.01, 0.01, 0.01])) = 0.1
        assert rmsd == pytest.approx(0.1)

    def test_shape_mismatch_raises_error(self):
        """Should raise ValueError for mismatched shapes."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="same shape"):
            calculate_rmsd(coords1, coords2)

    def test_invalid_shape_raises_error(self):
        """Should raise ValueError for non-Nx3 arrays."""
        coords1 = np.array([[0.0, 0.0], [1.0, 0.0]])  # Nx2
        coords2 = np.array([[0.0, 0.0], [1.0, 0.0]])

        with pytest.raises(ValueError, match="Nx3"):
            calculate_rmsd(coords1, coords2)

    def test_1d_array_raises_error(self):
        """Should raise ValueError for 1D arrays."""
        coords1 = np.array([0.0, 0.0, 0.0])
        coords2 = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Nx3"):
            calculate_rmsd(coords1, coords2)


class TestCalculatePairwiseRMSD:
    """Test calculate_pairwise_rmsd function."""

    def test_two_identical_structures(self):
        """Should return zero RMSD for identical structures."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        rmsd_matrix = calculate_pairwise_rmsd([coords1, coords2])

        assert rmsd_matrix.shape == (2, 2)
        assert rmsd_matrix[0, 0] == pytest.approx(0.0)
        assert rmsd_matrix[1, 1] == pytest.approx(0.0)
        assert rmsd_matrix[0, 1] == pytest.approx(0.0)
        assert rmsd_matrix[1, 0] == pytest.approx(0.0)

    def test_three_structures(self):
        """Should calculate pairwise RMSD for three structures."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords3 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        rmsd_matrix = calculate_pairwise_rmsd([coords1, coords2, coords3])

        assert rmsd_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert rmsd_matrix[0, 0] == pytest.approx(0.0)
        assert rmsd_matrix[1, 1] == pytest.approx(0.0)
        assert rmsd_matrix[2, 2] == pytest.approx(0.0)
        # coords1 vs coords2 should be 0
        assert rmsd_matrix[0, 1] == pytest.approx(0.0)
        # coords1 vs coords3 should be 1.0
        assert rmsd_matrix[0, 2] == pytest.approx(1.0)

    def test_symmetry(self):
        """Should return symmetric matrix."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])
        coords3 = np.array([[0.0, 1.0, 0.0]])

        rmsd_matrix = calculate_pairwise_rmsd([coords1, coords2, coords3])

        # Check symmetry
        assert rmsd_matrix[0, 1] == pytest.approx(rmsd_matrix[1, 0])
        assert rmsd_matrix[0, 2] == pytest.approx(rmsd_matrix[2, 0])
        assert rmsd_matrix[1, 2] == pytest.approx(rmsd_matrix[2, 1])

    def test_single_structure(self):
        """Should handle single structure (1x1 matrix)."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        rmsd_matrix = calculate_pairwise_rmsd([coords])

        assert rmsd_matrix.shape == (1, 1)
        assert rmsd_matrix[0, 0] == pytest.approx(0.0)

    def test_known_values(self):
        """Should calculate correct pairwise RMSD values."""
        # Create structures with known distances
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[3.0, 4.0, 0.0]])  # Distance = 5
        coords3 = np.array([[6.0, 8.0, 0.0]])  # Distance from 1 = 10, from 2 = 5

        rmsd_matrix = calculate_pairwise_rmsd([coords1, coords2, coords3])

        assert rmsd_matrix[0, 1] == pytest.approx(5.0)
        assert rmsd_matrix[0, 2] == pytest.approx(10.0)
        assert rmsd_matrix[1, 2] == pytest.approx(5.0)


class TestCalculateAverageCoords:
    """Test calculate_average_coords function."""

    def test_two_structures(self):
        """Should calculate average of two structures."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        avg = calculate_average_coords([coords1, coords2])

        expected = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert np.allclose(avg, expected)

    def test_three_structures(self):
        """Should calculate average of three structures."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[3.0, 0.0, 0.0]])
        coords3 = np.array([[6.0, 0.0, 0.0]])

        avg = calculate_average_coords([coords1, coords2, coords3])

        expected = np.array([[3.0, 0.0, 0.0]])
        assert np.allclose(avg, expected)

    def test_identical_structures(self):
        """Should return same structure for identical input."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        avg = calculate_average_coords([coords, coords, coords])

        assert np.allclose(avg, coords)

    def test_empty_list(self):
        """Should return empty array for empty input."""
        avg = calculate_average_coords([])

        assert avg.shape == (0, 3)

    def test_single_structure(self):
        """Should return same structure for single input."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        avg = calculate_average_coords([coords])

        assert np.allclose(avg, coords)

    def test_different_positions(self):
        """Should average different atom positions."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])

        avg = calculate_average_coords([coords1, coords2])

        expected = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        assert np.allclose(avg, expected)


class TestCalculateRMSDToAverage:
    """Test calculate_rmsd_to_average function."""

    def test_identical_structures(self):
        """Should return 0.0 RMSD for identical structures."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords, coords])

        assert avg_rmsd == pytest.approx(0.0)
        assert np.allclose(avg_coords, coords)

    def test_two_structures(self):
        """Should calculate RMSD to average for two structures."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[2.0, 0.0, 0.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords1, coords2])

        # Average is at [1, 0, 0]
        # RMSD from coords1 to avg = 1.0
        # RMSD from coords2 to avg = 1.0
        # Average RMSD = 1.0
        assert avg_rmsd == pytest.approx(1.0)
        assert np.allclose(avg_coords, [[1.0, 0.0, 0.0]])

    def test_three_structures_symmetric(self):
        """Should calculate RMSD for symmetric distribution."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])
        coords3 = np.array([[2.0, 0.0, 0.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords1, coords2, coords3])

        # Average is at [1, 0, 0]
        # RMSD: coords1→avg = 1, coords2→avg = 0, coords3→avg = 1
        # Average RMSD = (1 + 0 + 1) / 3 = 2/3
        expected_rmsd = 2.0 / 3.0
        assert avg_rmsd == pytest.approx(expected_rmsd)
        assert np.allclose(avg_coords, [[1.0, 0.0, 0.0]])

    def test_multiple_atoms(self):
        """Should handle multiple atoms per structure."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords1, coords2])

        # Average coords: [[0, 0.5, 0], [1, 0.5, 0]]
        expected_avg = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
        assert np.allclose(avg_coords, expected_avg)

        # Each structure is 0.5 away in y-direction
        # RMSD for each = sqrt(mean([0.25, 0.25])) = 0.5
        assert avg_rmsd == pytest.approx(0.5)

    def test_single_structure(self):
        """Should return 0.0 RMSD for single structure."""
        coords = np.array([[1.0, 2.0, 3.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords])

        assert avg_rmsd == pytest.approx(0.0)
        assert np.allclose(avg_coords, coords)


class TestRMSDIntegration:
    """Integration tests combining multiple RMSD functions."""

    def test_pairwise_vs_individual(self):
        """Pairwise matrix should match individual calculations."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        coords3 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        # Calculate pairwise
        rmsd_matrix = calculate_pairwise_rmsd([coords1, coords2, coords3])

        # Calculate individually
        rmsd_12 = calculate_rmsd(coords1, coords2)
        rmsd_13 = calculate_rmsd(coords1, coords3)
        rmsd_23 = calculate_rmsd(coords2, coords3)

        assert rmsd_matrix[0, 1] == pytest.approx(rmsd_12)
        assert rmsd_matrix[0, 2] == pytest.approx(rmsd_13)
        assert rmsd_matrix[1, 2] == pytest.approx(rmsd_23)

    def test_average_rmsd_consistency(self):
        """Average RMSD should be consistent with manual calculation."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])
        coords3 = np.array([[2.0, 0.0, 0.0]])

        avg_rmsd, avg_coords = calculate_rmsd_to_average([coords1, coords2, coords3])

        # Manual calculation
        manual_avg = calculate_average_coords([coords1, coords2, coords3])
        rmsd1 = calculate_rmsd(coords1, manual_avg)
        rmsd2 = calculate_rmsd(coords2, manual_avg)
        rmsd3 = calculate_rmsd(coords3, manual_avg)
        manual_avg_rmsd = np.mean([rmsd1, rmsd2, rmsd3])

        assert avg_rmsd == pytest.approx(manual_avg_rmsd)
        assert np.allclose(avg_coords, manual_avg)

    def test_ensemble_statistics(self):
        """Should calculate ensemble statistics correctly."""
        # Create a simple ensemble
        ensemble = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, -0.1, 0.0]]),
        ]

        # Calculate average structure
        avg_coords = calculate_average_coords(ensemble)
        assert avg_coords.shape == (2, 3)

        # Calculate RMSD to average
        avg_rmsd, _ = calculate_rmsd_to_average(ensemble)
        assert avg_rmsd > 0  # Should have some deviation

        # Calculate pairwise RMSD
        rmsd_matrix = calculate_pairwise_rmsd(ensemble)
        assert rmsd_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert np.allclose(np.diag(rmsd_matrix), 0)

    def test_large_ensemble(self):
        """Should handle larger ensembles efficiently."""
        # Create 10-structure ensemble
        ensemble = []
        for i in range(10):
            coords = np.array([[i * 0.1, 0.0, 0.0], [1 + i * 0.1, 0.0, 0.0]])
            ensemble.append(coords)

        # All functions should work
        avg_coords = calculate_average_coords(ensemble)
        assert avg_coords.shape == (2, 3)

        avg_rmsd, _ = calculate_rmsd_to_average(ensemble)
        assert avg_rmsd >= 0

        rmsd_matrix = calculate_pairwise_rmsd(ensemble)
        assert rmsd_matrix.shape == (10, 10)
        assert np.allclose(np.diag(rmsd_matrix), 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_rmsd(self):
        """Should handle very small RMSD values."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1e-10, 1e-10, 1e-10]])

        rmsd = calculate_rmsd(coords1, coords2)

        assert rmsd < 1e-9
        assert rmsd >= 0

    def test_very_large_rmsd(self):
        """Should handle very large RMSD values."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1e6, 1e6, 1e6]])

        rmsd = calculate_rmsd(coords1, coords2)

        expected = np.sqrt(3) * 1e6
        assert rmsd == pytest.approx(expected, rel=1e-9)

    def test_negative_coordinates(self):
        """Should handle negative coordinates."""
        coords1 = np.array([[-1.0, -2.0, -3.0]])
        coords2 = np.array([[-1.5, -2.5, -3.5]])

        rmsd = calculate_rmsd(coords1, coords2)

        expected = np.sqrt(0.75)
        assert rmsd == pytest.approx(expected)

    def test_mixed_sign_coordinates(self):
        """Should handle mixed positive/negative coordinates."""
        coords1 = np.array([[1.0, -1.0, 0.0]])
        coords2 = np.array([[-1.0, 1.0, 0.0]])

        rmsd = calculate_rmsd(coords1, coords2)

        # Distance = sqrt(4 + 4 + 0) = sqrt(8)
        assert rmsd == pytest.approx(np.sqrt(8))
