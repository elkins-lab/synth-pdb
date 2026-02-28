"""
Tests targeting geometry.py coverage gaps:
  - njit fallback decorator (lines 9-17)
  - calculate_angle (lines 257-278)
  - calculate_dihedral_angle (lines 281-326)
  - batched_angle / batched_dihedral
  - reconstruct_sidechain rotation path (rotate_points inner fn + superimpose)
"""
import math
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_close(a: float, b: float, abs_tol: float = 0.5) -> None:
    assert abs(a - b) < abs_tol, f"{a} != {b} (tol={abs_tol})"


# ---------------------------------------------------------------------------
# njit fallback decorator (lines 9-17 in geometry.py)
# ---------------------------------------------------------------------------

class TestNjitFallback:
    """
    Confirm that the no-op njit fallback defined when numba is absent
    passes functions through unchanged.
    """

    def test_njit_passthrough_direct(self, monkeypatch):
        """When numba is absent the fallback njit must act as identity."""
        import sys
        # Simulate numba absence by temporarily hiding it
        original_numba = sys.modules.get("numba")
        sys.modules["numba"] = None  # type: ignore[assignment]

        try:
            # Re-import geometry in a fresh module to trigger the ImportError branch
            import importlib
            import synth_pdb.geometry as geom_mod
            importlib.reload(geom_mod)

            @geom_mod.njit  # type: ignore[misc]
            def add_one(x: float) -> float:
                return x + 1.0

            assert add_one(4.0) == 5.0, "njit fallback must return a callable function"
        finally:
            # Restore numba (or remove the sentinel)
            if original_numba is None:
                sys.modules.pop("numba", None)
            else:
                sys.modules["numba"] = original_numba

    def test_njit_fallback_with_kwargs(self):
        """Fallback must also handle njit(cache=True) call-style."""
        # Import geometry normally and check that the decorator is callable
        from synth_pdb.geometry import njit  # type: ignore[attr-defined]

        @njit(cache=True)  # type: ignore[misc]
        def mul_two(x: float) -> float:
            return x * 2.0

        assert mul_two(3.0) == 6.0


# ---------------------------------------------------------------------------
# calculate_angle  (lines 257-278)
# ---------------------------------------------------------------------------

class TestCalculateAngle:
    """Tests for the @njit calculate_angle function."""

    def test_right_angle(self):
        from synth_pdb.geometry import calculate_angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(calculate_angle(p1, p2, p3), abs=1e-4) == 90.0

    def test_straight_angle(self):
        from synth_pdb.geometry import calculate_angle
        p1 = np.array([-1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        assert pytest.approx(calculate_angle(p1, p2, p3), abs=1e-4) == 180.0

    def test_60_degree_angle(self):
        from synth_pdb.geometry import calculate_angle
        # Equilateral triangle vertex
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.5, math.sqrt(3) / 2, 0.0])
        angle = calculate_angle(p1, p2, p3)
        assert pytest.approx(angle, abs=1e-3) == 60.0

    def test_degenerate_zero_denominator_returns_zero(self):
        """If two vectors are zero-length, result should not raise (returns 0.0)."""
        from synth_pdb.geometry import calculate_angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])  # same as p1 → zero vector
        p3 = np.array([1.0, 0.0, 0.0])
        result = calculate_angle(p1, p2, p3)
        assert result == 0.0


# ---------------------------------------------------------------------------
# calculate_dihedral_angle  (lines 281-326)
# ---------------------------------------------------------------------------

class TestCalculateDihedralAngle:
    """Tests for the @njit calculate_dihedral_angle function."""

    def test_180_degree_trans_dihedral(self):
        from synth_pdb.geometry import calculate_dihedral_angle
        # Classic trans arrangement
        p1 = np.array([0.0,  1.0, 0.0])
        p2 = np.array([0.0,  0.0, 0.0])
        p3 = np.array([1.0,  0.0, 0.0])
        p4 = np.array([1.0, -1.0, 0.0])
        dihedral = calculate_dihedral_angle(p1, p2, p3, p4)
        assert pytest.approx(abs(dihedral), abs=1.0) == 180.0

    def test_0_degree_cis_dihedral(self):
        from synth_pdb.geometry import calculate_dihedral_angle
        # All four atoms in the same plane, same-side
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        dihedral = calculate_dihedral_angle(p1, p2, p3, p4)
        assert pytest.approx(abs(dihedral), abs=1.0) == 0.0

    def test_90_degree_dihedral(self):
        from synth_pdb.geometry import calculate_dihedral_angle
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([0.0, 0.0, 1.0])
        dihedral = calculate_dihedral_angle(p1, p2, p3, p4)
        assert pytest.approx(abs(dihedral), abs=1.0) == 90.0

    def test_dihedral_degenerate_zero_vector(self):
        """Degenerate input (p2 == p3) should not raise."""
        from synth_pdb.geometry import calculate_dihedral_angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])  # same as p2
        p4 = np.array([0.0, 1.0, 0.0])
        # Should return a float (may be 0.0 or ±180 deg) without crashing
        result = calculate_dihedral_angle(p1, p2, p3, p4)
        assert isinstance(float(result), float)


# ---------------------------------------------------------------------------
# batched_angle / batched_dihedral
# ---------------------------------------------------------------------------

class TestBatchedGeometry:
    """Tests for vectorised angle helpers."""

    def test_batched_angle_right_angles(self):
        from synth_pdb.geometry import batched_angle
        # Three sets of right-angle triplets
        n = 5
        p1 = np.tile([1.0, 0.0, 0.0], (n, 1))
        p2 = np.zeros((n, 3))
        p3 = np.tile([0.0, 1.0, 0.0], (n, 1))
        angles = batched_angle(p1, p2, p3)
        assert angles.shape == (n,)
        np.testing.assert_allclose(angles, 90.0, atol=1e-4)

    def test_batched_angle_straight_line(self):
        from synth_pdb.geometry import batched_angle
        p1 = np.array([[-1.0, 0.0, 0.0]])
        p2 = np.zeros((1, 3))
        p3 = np.array([[1.0, 0.0, 0.0]])
        angles = batched_angle(p1, p2, p3)
        # batched_angle has a small numeric stabiliser (+1e-9), so exact 180°
        # may not be achieved; accept within 0.1°
        np.testing.assert_allclose(angles, 180.0, atol=0.1)

    def test_batched_dihedral_trans(self):
        from synth_pdb.geometry import batched_dihedral
        # Single trans dihedral
        p1 = np.array([[0.0,  1.0, 0.0]])
        p2 = np.array([[0.0,  0.0, 0.0]])
        p3 = np.array([[1.0,  0.0, 0.0]])
        p4 = np.array([[1.0, -1.0, 0.0]])
        dihedrals = batched_dihedral(p1, p2, p3, p4)
        assert dihedrals.shape == (1,)
        assert pytest.approx(abs(float(dihedrals[0])), abs=1.0) == 180.0

    def test_batched_dihedral_batch_consistency(self):
        """Batched result must match scalar result for random inputs."""
        from synth_pdb.geometry import batched_dihedral, calculate_dihedral_angle
        rng = np.random.default_rng(42)
        p1 = rng.standard_normal((8, 3))
        p2 = rng.standard_normal((8, 3))
        p3 = rng.standard_normal((8, 3))
        p4 = rng.standard_normal((8, 3))
        batch_result = batched_dihedral(p1, p2, p3, p4)
        for i in range(8):
            scalar = calculate_dihedral_angle(p1[i], p2[i], p3[i], p4[i])
            assert pytest.approx(float(batch_result[i]), abs=0.5) == float(scalar)


# ---------------------------------------------------------------------------
# reconstruct_sidechain — rotation path (rotate_points + superimpose branch)
# ---------------------------------------------------------------------------

class TestReconstructSidechainRotation:
    """
    Specifically target the sidechain-rotation branch inside reconstruct_sidechain
    (lines 525-605 in geometry.py).  We use residues with gamma atoms so the
    sidechain-rotation code path is exercised.
    """

    @pytest.fixture(scope="class")
    def pdb_with_phe(self):
        from synth_pdb.generator import generate_pdb_content
        return generate_pdb_content(sequence_str="AFKA", conformation="alpha",
                                    minimize_energy=False)

    def test_rotation_path_does_not_raise(self, pdb_with_phe):
        """Calling reconstruct_sidechain on PHE should exercise the rotate_points path."""
        import io
        import biotite.structure.io.pdb as bpdb
        from synth_pdb.geometry import reconstruct_sidechain
        pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_with_phe))
        peptide = pdb_file.get_structure(model=1)
        # PHE is res_id 2 (second residue, A=1, F=2)
        reconstruct_sidechain(peptide, 2, {"chi1": [180.0]})

    def test_rotation_path_changes_sidechain(self, pdb_with_phe):
        """Sidechain atoms of PHE must move after applying a different chi1."""
        import io
        import biotite.structure.io.pdb as bpdb
        from synth_pdb.geometry import reconstruct_sidechain
        pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_with_phe))
        peptide = pdb_file.get_structure(model=1)

        res_mask = peptide.res_id == 2
        orig = peptide.coord[res_mask].copy()
        reconstruct_sidechain(peptide, 2, {"chi1": [-60.0]})
        updated = peptide.coord[res_mask]
        # At least some atoms must have moved
        assert not np.allclose(orig, updated, atol=0.01), (
            "No atoms changed after reconstruct_sidechain on PHE"
        )

    def test_rotation_path_lys_long_chain(self):
        """LYS has chi1-chi4; exercise the rotation path with a long sidechain."""
        import io
        import biotite.structure.io.pdb as bpdb
        from synth_pdb.generator import generate_pdb_content
        from synth_pdb.geometry import reconstruct_sidechain
        pdb_content = generate_pdb_content(sequence_str="AKA", conformation="alpha",
                                           minimize_energy=False)
        pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_content))
        peptide = pdb_file.get_structure(model=1)
        # Should not raise even for a deep sidechain
        reconstruct_sidechain(peptide, 2, {"chi1": [60.0]})

    def test_reconstruct_with_scalar_rotamer_entry(self):
        """
        RotamerEntry values may be Union[float, List[float]].
        Passing a plain float for chi1 should be handled gracefully.
        """
        import io
        import biotite.structure.io.pdb as bpdb
        from synth_pdb.generator import generate_pdb_content
        from synth_pdb.geometry import reconstruct_sidechain
        pdb_content = generate_pdb_content(sequence_str="AC", conformation="alpha",
                                           minimize_energy=False)
        pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_content))
        peptide = pdb_file.get_structure(model=1)
        # Pass plain float (not wrapped in list) — exercises the Union branch
        reconstruct_sidechain(peptide, 2, {"chi1": 60.0, "prob": 0.9})


# ---------------------------------------------------------------------------
# position_atom_3d_from_internal_coords — edge cases
# ---------------------------------------------------------------------------

class TestPositionAtom3d:
    """Additional coverage for the NeRF atom placement function."""

    def test_bond_length_is_respected(self):
        from synth_pdb.geometry import position_atom_3d_from_internal_coords
        # Use non-collinear geometry to avoid degenerate cross-products
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([1.5, 0.0, 0.0])
        p3 = np.array([2.0, 1.0, 0.0])
        target_len = 1.52
        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, target_len, 120.0, 180.0)
        actual_len = float(np.linalg.norm(p4 - p3))
        assert pytest.approx(actual_len, abs=0.01) == target_len

    def test_different_dihedral_angles_produce_different_positions(self):
        from synth_pdb.geometry import position_atom_3d_from_internal_coords
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([1.5, 0.0, 0.0])
        p3 = np.array([2.0, 1.0, 0.0])
        pos_60  = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0,  60.0)
        pos_180 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, 180.0)
        pos_300 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, 300.0)
        assert not np.allclose(pos_60, pos_180)
        assert not np.allclose(pos_60, pos_300)
        assert not np.allclose(pos_180, pos_300)

    def test_output_is_3d_float_array(self):
        from synth_pdb.geometry import position_atom_3d_from_internal_coords
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.5, 0.0, 0.0])
        p3 = np.array([2.0, 1.0, 0.0])
        p4 = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, -60.0)
        assert p4.shape == (3,)
        assert p4.dtype in (np.float32, np.float64)
        assert not np.any(np.isnan(p4))
