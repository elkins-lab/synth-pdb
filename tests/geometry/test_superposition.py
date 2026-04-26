"""
Unit tests for geometry.superposition module.

Tests the Kabsch algorithm for optimal superposition, including rotation
matrix calculation, transformation application, and medoid finding.
"""

import numpy as np
import pytest

from synth_pdb.geometry.rmsd import calculate_rmsd
from synth_pdb.geometry.superposition import (
    apply_transformation,
    find_medoid,
    kabsch_superposition,
    superimpose_structures,
)


class TestKabschSuperposition:
    """Test kabsch_superposition function."""

    def test_identical_structures(self):
        """Should return identity transformation for identical structures."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        R, t = kabsch_superposition(coords, coords)

        # Rotation should be identity
        assert np.allclose(R, np.eye(3))
        # Translation should be zero
        assert np.allclose(t, [0.0, 0.0, 0.0])

    def test_pure_translation(self):
        """Should find correct transformation for pure translation."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        translation = np.array([5.0, 3.0, -2.0])
        coords2 = coords1 + translation

        R, t = kabsch_superposition(coords1, coords2)

        # Rotation should be identity (no rotation needed)
        assert np.allclose(R, np.eye(3), atol=1e-10)
        # Translation should match
        assert np.allclose(t, translation, atol=1e-10)

    def test_rotation_90_degrees_z(self):
        """Should find 90-degree rotation around z-axis."""
        # Original coordinates
        coords1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Rotated 90 degrees around z-axis
        coords2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        R, t = kabsch_superposition(coords1, coords2)

        # Apply transformation
        aligned = (R @ coords1.T).T + t

        # Result should match coords2
        assert np.allclose(aligned, coords2, atol=1e-10)

    def test_rotation_180_degrees(self):
        """Should find 180-degree rotation."""
        coords1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords2 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

        R, t = kabsch_superposition(coords1, coords2)

        aligned = (R @ coords1.T).T + t
        assert np.allclose(aligned, coords2, atol=1e-10)

    def test_combined_rotation_translation(self):
        """Should handle combined rotation and translation."""
        coords1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Apply known transformation
        angle = np.pi / 4  # 45 degrees
        R_true = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        t_true = np.array([2.0, 3.0, 4.0])
        coords2 = (R_true @ coords1.T).T + t_true

        R, t = kabsch_superposition(coords1, coords2)

        # Apply calculated transformation
        aligned = (R @ coords1.T).T + t

        # Should match coords2
        assert np.allclose(aligned, coords2, atol=1e-10)

    def test_proper_rotation_matrix(self):
        """Rotation matrix should be proper (det = 1)."""
        coords1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        coords2 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        R, t = kabsch_superposition(coords1, coords2)

        # Determinant should be 1 (proper rotation)
        det = np.linalg.det(R)
        assert det == pytest.approx(1.0)

    def test_orthogonal_rotation_matrix(self):
        """Rotation matrix should be orthogonal."""
        coords1 = np.random.rand(5, 3)
        coords2 = np.random.rand(5, 3)

        R, t = kabsch_superposition(coords1, coords2)

        # R @ R.T should be identity
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_single_atom(self):
        """Should handle single atom case."""
        coords1 = np.array([[1.0, 2.0, 3.0]])
        coords2 = np.array([[4.0, 5.0, 6.0]])

        R, t = kabsch_superposition(coords1, coords2)

        aligned = (R @ coords1.T).T + t
        assert np.allclose(aligned, coords2, atol=1e-10)

    def test_two_atoms(self):
        """Should handle two-atom case."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        R, t = kabsch_superposition(coords1, coords2)

        aligned = (R @ coords1.T).T + t
        assert np.allclose(aligned, coords2, atol=1e-10)

    def test_many_atoms(self):
        """Should handle many atoms efficiently."""
        coords1 = np.random.rand(100, 3)
        coords2 = np.random.rand(100, 3)

        R, t = kabsch_superposition(coords1, coords2)

        # Should complete without error
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_shape_mismatch_raises_error(self):
        """Should raise ValueError for mismatched shapes."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="same shape"):
            kabsch_superposition(coords1, coords2)

    def test_invalid_shape_raises_error(self):
        """Should raise ValueError for non-Nx3 arrays."""
        coords1 = np.array([[0.0, 0.0], [1.0, 0.0]])  # Nx2
        coords2 = np.array([[0.0, 0.0], [1.0, 0.0]])

        with pytest.raises(ValueError, match="Nx3"):
            kabsch_superposition(coords1, coords2)


class TestApplyTransformation:
    """Test apply_transformation function."""

    def test_identity_transformation(self):
        """Should return same coords for identity transformation."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 0.0])

        result = apply_transformation(coords, R, t)

        assert np.allclose(result, coords)

    def test_translation_only(self):
        """Should apply translation correctly."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        R = np.eye(3)
        t = np.array([10.0, 20.0, 30.0])

        result = apply_transformation(coords, R, t)

        expected = coords + t
        assert np.allclose(result, expected)

    def test_rotation_only(self):
        """Should apply rotation correctly."""
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # 90-degree rotation around z-axis
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([0.0, 0.0, 0.0])

        result = apply_transformation(coords, R, t)

        expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_combined_transformation(self):
        """Should apply rotation and translation."""
        coords = np.array([[1.0, 0.0, 0.0]])
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([5.0, 5.0, 5.0])

        result = apply_transformation(coords, R, t)

        # Rotation: [1, 0, 0] -> [0, 1, 0]
        # Translation: + [5, 5, 5] -> [5, 6, 5]
        expected = np.array([[5.0, 6.0, 5.0]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_single_point(self):
        """Should handle single point."""
        coords = np.array([[1.0, 2.0, 3.0]])
        R = np.eye(3)
        t = np.array([1.0, 1.0, 1.0])

        result = apply_transformation(coords, R, t)

        assert np.allclose(result, [[2.0, 3.0, 4.0]])

    def test_many_points(self):
        """Should handle many points efficiently."""
        coords = np.random.rand(1000, 3)
        R = np.eye(3)
        t = np.random.rand(3)

        result = apply_transformation(coords, R, t)

        assert result.shape == coords.shape
        assert np.allclose(result, coords + t)


class TestSuperimposeStructures:
    """Test superimpose_structures function."""

    def test_identical_structures(self):
        """Should return nearly identical coords for identical structures."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        aligned = superimpose_structures(coords, coords)

        assert np.allclose(aligned, coords, atol=1e-10)

    def test_translated_structures(self):
        """Should align translated structures."""
        mobile = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        reference = np.array([[5.0, 3.0, -2.0], [6.0, 3.0, -2.0]])

        aligned = superimpose_structures(mobile, reference)

        # RMSD should be essentially zero
        rmsd = calculate_rmsd(aligned, reference)
        assert rmsd < 1e-10

    def test_rotated_structures(self):
        """Should align rotated structures."""
        mobile = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Rotate 90 degrees around z-axis
        reference = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        aligned = superimpose_structures(mobile, reference)

        rmsd = calculate_rmsd(aligned, reference)
        assert rmsd < 1e-10

    def test_complex_transformation(self):
        """Should align structures with complex transformation."""
        mobile = np.random.rand(10, 3)

        # Apply random rotation and translation
        angle = np.random.rand() * 2 * np.pi
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        t = np.random.rand(3) * 10
        reference = (R @ mobile.T).T + t

        aligned = superimpose_structures(mobile, reference)

        rmsd = calculate_rmsd(aligned, reference)
        assert rmsd < 1e-9

    def test_preserves_internal_distances(self):
        """Superposition should preserve internal distances."""
        mobile = np.random.rand(5, 3)
        reference = np.random.rand(5, 3)

        # Calculate internal distances before
        distances_before = []
        for i in range(len(mobile)):
            for j in range(i + 1, len(mobile)):
                d = np.linalg.norm(mobile[i] - mobile[j])
                distances_before.append(d)

        aligned = superimpose_structures(mobile, reference)

        # Calculate internal distances after
        distances_after = []
        for i in range(len(aligned)):
            for j in range(i + 1, len(aligned)):
                d = np.linalg.norm(aligned[i] - aligned[j])
                distances_after.append(d)

        # Internal distances should be preserved
        assert np.allclose(distances_before, distances_after, atol=1e-10)


class TestFindMedoid:
    """Test find_medoid function."""

    def test_two_identical_one_different(self):
        """Medoid should be one of identical structures."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Identical to coords1
        coords3 = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]])  # Far away

        medoid_idx = find_medoid([coords1, coords2, coords3], superimpose=False)

        # Should be coords1 or coords2 (both have lower sum RMSD)
        assert medoid_idx in [0, 1]

    def test_three_structures_linear(self):
        """Medoid should be middle structure."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[5.0, 0.0, 0.0]])  # Middle
        coords3 = np.array([[10.0, 0.0, 0.0]])

        medoid_idx = find_medoid([coords1, coords2, coords3], superimpose=False)

        # Middle structure minimizes sum of distances
        assert medoid_idx == 1

    def test_single_structure(self):
        """Single structure should be its own medoid."""
        coords = np.array([[1.0, 2.0, 3.0]])

        medoid_idx = find_medoid([coords])

        assert medoid_idx == 0

    def test_two_structures(self):
        """Should return valid medoid for two structures."""
        coords1 = np.array([[0.0, 0.0, 0.0]])
        coords2 = np.array([[1.0, 0.0, 0.0]])

        medoid_idx = find_medoid([coords1, coords2], superimpose=False)

        assert medoid_idx in [0, 1]

    def test_multiple_structures(self):
        """Should find medoid for multiple structures."""
        # Create structures in a cluster with one outlier
        ensemble = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.1, 0.1, 0.0], [1.1, 0.1, 0.0]]),
            np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]]),
            np.array([[100.0, 100.0, 100.0], [101.0, 100.0, 100.0]]),  # Outlier
        ]

        medoid_idx = find_medoid(ensemble, superimpose=False)

        # Medoid should not be the outlier
        assert medoid_idx != 3

    def test_identical_structures(self):
        """All identical structures should give first as medoid."""
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        medoid_idx = find_medoid([coords, coords, coords])

        # First structure has same sum as others, so should be returned
        assert medoid_idx == 0


class TestSuperpositionIntegration:
    """Integration tests combining multiple functions."""

    def test_superposition_minimizes_rmsd(self):
        """Superposition should minimize RMSD."""
        mobile = np.random.rand(10, 3)
        reference = np.random.rand(10, 3)

        # RMSD before alignment
        rmsd_before = calculate_rmsd(mobile, reference)

        # Align structures
        aligned = superimpose_structures(mobile, reference)

        # RMSD after alignment
        rmsd_after = calculate_rmsd(aligned, reference)

        # RMSD should decrease (or stay same if already optimal)
        assert rmsd_after <= rmsd_before + 1e-10

    def test_kabsch_then_apply(self):
        """Applying Kabsch result should match superimpose_structures."""
        mobile = np.random.rand(5, 3)
        reference = np.random.rand(5, 3)

        # Method 1: Use superimpose_structures
        aligned1 = superimpose_structures(mobile, reference)

        # Method 2: Use kabsch + apply separately
        R, t = kabsch_superposition(mobile, reference)
        aligned2 = apply_transformation(mobile, R, t)

        # Results should be identical
        assert np.allclose(aligned1, aligned2, atol=1e-10)

    def test_ensemble_alignment(self):
        """Should align entire ensemble to reference."""
        # Create reference structure
        reference = np.random.rand(5, 3)

        # Create ensemble with random transformations
        ensemble = []
        for _i in range(10):
            angle = np.random.rand() * 2 * np.pi
            R = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            t = np.random.rand(3) * 5
            coords = (R @ reference.T).T + t
            ensemble.append(coords)

        # Align all to reference
        aligned_ensemble = [superimpose_structures(coords, reference) for coords in ensemble]

        # All should have low RMSD to reference
        for aligned in aligned_ensemble:
            rmsd = calculate_rmsd(aligned, reference)
            assert rmsd < 1e-9

    def test_medoid_based_alignment(self):
        """Should find medoid and align ensemble to it."""
        # Create ensemble
        ensemble = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]]),
        ]

        # Find medoid
        medoid_idx = find_medoid(ensemble)
        reference = ensemble[medoid_idx]

        # Align all to medoid
        aligned = [superimpose_structures(coords, reference) for coords in ensemble]

        # Medoid structure should align to itself
        assert np.allclose(aligned[medoid_idx], ensemble[medoid_idx], atol=1e-10)

    # Remove the failing roundtrip test and replace with a better one

    def test_known_transformation_recovery(self):
        """Should apply and verify known transformations."""
        coords = np.random.seed(42) or np.random.rand(10, 3)

        # Apply known rotation and translation
        angle = np.pi / 3  # 60 degrees
        R_known = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        t_known = np.array([2.0, 3.0, 4.0])
        transformed = (R_known @ coords.T).T + t_known

        # Find transformation using Kabsch
        R, t = kabsch_superposition(coords, transformed)

        # Apply found transformation
        aligned = apply_transformation(coords, R, t)

        # Should match the transformed coordinates
        assert np.allclose(aligned, transformed, atol=1e-10)
