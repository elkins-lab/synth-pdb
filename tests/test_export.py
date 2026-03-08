import numpy as np
import pytest

from synth_pdb.export import export_constraints


@pytest.fixture
def mock_matrix():
    """Create a 4x4 matrix representing 4 residues."""
    # Res 1-4
    # Contacts: (1, 4) is Long Range. (1, 2) is neighbor.
    mat = np.zeros((4, 4))
    mat[0, 3] = 1.0  # Contact between 1 and 4
    mat[3, 0] = 1.0

    mat[0, 1] = 1.0  # Neighbor
    mat[1, 0] = 1.0
    return mat


def test_export_casp_format(mock_matrix):
    """Test exporting to CASP RR format."""
    # Sequence: A A A A
    seq = "AAAA"

    output = export_constraints(mock_matrix, sequence=seq, fmt="casp")

    lines = output.strip().split("\n")
    # Header should perform sequence
    assert lines[0] == seq

    # Body: i j d1 d2 p
    # Should contain 1 4 0 8 1.0
    # Neighbors (1 2) might be filtered or included depending on logic.
    # Standard CASP usually wants separation >= 6.
    # But let's assume we output ALL for now defined by the matrix.

    # Finding the line for 1-4 contact (Indices 0, 3 -> ResIDs 1, 4)
    found = False
    for line in lines:
        if line.startswith("1 4"):
            parts = line.split()
            assert parts[0] == "1"
            assert parts[1] == "4"
            assert float(parts[2]) == 0.0  # d_min
            assert float(parts[3]) == 8.0  # d_max (Standard contact threshold)
            assert float(parts[4]) == 1.0  # Probability
            found = True

    assert found


def test_export_csv_format(mock_matrix):
    """Test simple CSV export."""
    seq = "AAAA"
    output = export_constraints(mock_matrix, sequence=seq, fmt="csv")

    # Expect: Res1,Res2,Value
    assert "1,4,1.0" in output


def test_export_casp_variety():
    """
    Regression test: Ensure CASP export reflects varying distances.
    This would have caught the issue where all lines had identical '8.0' thresholds.
    """
    # Create distance matrix with varied values > 1.0
    dist_mat = np.zeros((5, 5))
    dist_mat[0, 2] = 3.5
    dist_mat[0, 3] = 7.2
    dist_mat[1, 4] = 5.1

    seq = "AAAAA"
    # threshold 8.0
    output = export_constraints(dist_mat, sequence=seq, fmt="casp", threshold=8.0)

    lines = output.strip().split("\n")

    # Check for specific distance reflections in column 4 (d_max)
    # i j d_min d_max prob
    # 0 2 -> 1 3 -> 3.5
    # 0 3 -> 1 4 -> 7.2
    # 1 4 -> 2 5 -> 5.1

    assert "1 3 0.0 3.5 1.00000" in lines
    assert "1 4 0.0 7.2 1.00000" in lines
    assert "2 5 0.0 5.1 1.00000" in lines

    # Verify we don't have repetitive 8.0 values
    count_8 = sum(1 for line in lines if "8.0" in line)
    assert count_8 == 0, "CASP output should contain actual distances, not the default threshold."


def test_export_csv_variety():
    """Test CSV export with actual distances."""
    dist_mat = np.zeros((3, 3))
    dist_mat[0, 2] = 5.5

    seq = "AAA"
    output = export_constraints(dist_mat, sequence=seq, fmt="csv", threshold=6.0)

    # Header
    assert "Res1,Res2,Value" in output
    # Row
    assert "1,3,5.50000" in output


def test_export_invalid_format():
    """Test that unknown formats raise ValueError."""
    mat = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Unknown format: xml"):
        export_constraints(mat, "AA", fmt="xml")
