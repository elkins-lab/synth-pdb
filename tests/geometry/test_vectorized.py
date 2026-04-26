"""
Tests for vectorized geometry operations.
"""

import numpy as np
import pytest

from synth_pdb.geometry.vectorized import batched_angle, position_atoms_batch, superimpose_batch


def test_position_atoms_batch_basic():
    """Test the position_atoms_batch function with basic inputs."""
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    p3 = np.array([[1.0, 1.0, 0.0]])

    bond_lengths = np.array([1.0])
    bond_angles = np.array([90.0])
    dihedral_angles = np.array([0.0])

    p4 = position_atoms_batch(p1, p2, p3, bond_lengths, bond_angles, dihedral_angles)

    # Expected: p4 is [0, 1, 0] because it's 90 degrees from p3-p2-p1
    # Wait, let's re-calculate:
    # b = p3-p2 = [0, 1, 0]
    # a = p2-p1 = [1, 0, 0]
    # c = a x b = [0, 0, 1]
    # d = c x b = [-1, 0, 0]
    # p4 = p3 + 1.0 * (-b*cos(90) + d*sin(90)*cos(0) + c*sin(90)*sin(0))
    # p4 = p3 + 1.0 * (0 + d*1*1 + 0) = [1, 1, 0] + [-1, 0, 0] = [0, 1, 0]

    expected = np.array([[0.0, 1.0, 0.0]])
    assert np.allclose(p4, expected)


def test_superimpose_batch_reflection():
    """Test superimpose_batch with a reflection."""
    sources = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    reflection = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    targets = np.matmul(sources, reflection.T)

    trans, rot = superimpose_batch(sources, targets)
    assert np.allclose(np.linalg.det(rot), 1.0)


def test_batched_angle():
    """Test the batched_angle utility."""
    p1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p3 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # 90 deg and 0 deg

    angles = batched_angle(p1, p2, p3)
    # The second one might be slightly off due to epsilon in batched_angle
    assert angles[0] == pytest.approx(90.0)
    assert angles[1] == pytest.approx(0.0, abs=1e-2)
