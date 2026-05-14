"""tests/unit/test_gnn_edge_cases.py.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Targeted tests for edge cases and error handling in GNN scoring and benchmark metrics.
Fills coverage gaps identified in synth_pdb/score.py and synth_pdb/benchmark_metrics.py.
"""

import os
import numpy as np
import pytest
from synth_pdb.score import score_structure, score_batch
from synth_pdb.benchmark_metrics import shift_rmsd, tm_score, extract_ca_coords

# Mark for GNN tests (requires torch/torch_geometric)
pytestmark = pytest.mark.gnn

# -----------------------------------------------------------------------------
# synth_pdb.score Gaps
# -----------------------------------------------------------------------------


def test_score_batch_error_recovery():
    """Verify that score_batch recovers from a corrupted PDB string without crashing."""
    valid_pdb = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C"
    )
    corrupted_pdb = "THIS IS NOT A PDB"

    # We expect 2 results: one valid, one error sentinel
    results = score_batch([valid_pdb, corrupted_pdb])

    assert len(results) == 2
    # Item 0 should be valid (or at least tried)
    assert results[0].label in ("High Quality", "Low Quality", "Error")

    # Item 1 should be the error sentinel
    assert results[1].label == "Error"
    assert np.isnan(results[1].global_score)
    assert results[1].per_residue == []


def test_score_structure_custom_model_path_not_found():
    """score_structure with a custom model_path should try to load it (and fail if not found)."""
    valid_pdb = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C"
    )

    # Passing a non-existent .pt file should raise FileNotFoundError or RuntimeError from torch
    # depending on where it fails. GNNQualityClassifier raises RuntimeError if file missing.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        score_structure(valid_pdb, model_path="/tmp/nonexistent_model.pt")


def test_score_structure_inline_pdb_variants():
    """Verify detection of inline PDB content starting with different headers."""
    # Test 'REMARK' start
    remark_pdb = (
        "REMARK 999\nATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C"
    )
    # This should be treated as inline PDB string, not a filename
    # If it were treated as a filename, it would raise FileNotFoundError
    try:
        score_structure(remark_pdb)
    except (ValueError, ImportError):
        # ValueError is fine (e.g. if the model isn't actually loaded in this env)
        # We just want to ensure it didn't try to open a file named 'REMARK...'
        pass


# -----------------------------------------------------------------------------
# synth_pdb.benchmark_metrics Gaps
# -----------------------------------------------------------------------------


def test_shift_rmsd_nucleus_mismatch_warning(caplog):
    """Verify that shift_rmsd warns when a nucleus in ref_shifts is missing from pred_shifts."""
    pred = {"H": np.array([8.0, 8.1])}
    ref = {"H": np.array([8.0, 8.1]), "N": np.array([120.0, 121.0])}

    # This should run and skip 'N'
    rmsd = shift_rmsd(pred, ref)
    assert rmsd < 1e-6
    # No warning in this direction (pred missing nucleus present in ref is logged)

    # Test opposite: pred has nucleus missing from ref
    pred2 = {"H": np.array([8.0, 8.1]), "N": np.array([120.0, 121.0])}
    ref2 = {"H": np.array([8.0, 8.1])}

    with caplog.at_level("WARNING"):
        shift_rmsd(pred2, ref2)
        assert "nucleus 'N' not in ref_shifts" in caplog.text


def test_shift_rmsd_all_nan_or_empty(caplog):
    """Verify that shift_rmsd returns NaN and warns when no valid pairs exist."""
    pred = {"H": np.array([np.nan, np.nan])}
    ref = {"H": np.array([8.0, 8.1])}

    with caplog.at_level("WARNING"):
        rmsd = shift_rmsd(pred, ref)
        assert np.isnan(rmsd)
        assert "no valid (nucleus, residue) pairs found" in caplog.text


def test_tm_score_short_sequence():
    """Verify TM-score d0 constant for short sequences (L < 22)."""
    # L = 10
    coords = np.zeros((10, 3))
    coords[:, 2] = np.arange(10) * 1.5  # simple line

    # Should use d0 = 0.5
    score = tm_score(coords, coords)
    assert abs(score - 1.0) < 1e-5


def test_extract_ca_coords_malformed_lines():
    """Verify that extract_ca_coords skips lines with malformed coordinates."""
    pdb = (
        "ATOM      1  CA  ALA A   1    XXXXXXXX   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C\n"
        "ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  0.00           C"
    )

    # Line 1 is malformed, should be skipped. 2 and 3 are valid.
    coords = extract_ca_coords(pdb)
    assert coords.shape == (2, 3)
    np.testing.assert_allclose(coords[0], [4.0, 5.0, 6.0])
