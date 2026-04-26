"""
Tests for synth_pdb.ensemble.daop — Dihedral Angle Order Parameters.

Coverage target:
    - calculate_order_parameter  (unit + scientific validation)
    - find_well_defined_residues (unit + threshold boundary)
    - calculate_backbone_daop    (unit + shape consistency)
    - Public API accessible from synth_pdb.ensemble (import smoke test)
"""

import numpy as np
import pytest

from synth_pdb.ensemble.daop import DAOPCalculator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ordered_angles(n_models: int, center: float, sigma: float = 0.0) -> np.ndarray:
    """Return a 1-D array of n_models angles near center (radians)."""
    rng = np.random.default_rng(seed=42)
    if sigma == 0.0:
        return np.full(n_models, center)
    return rng.normal(center, sigma, n_models)


# ===========================================================================
# TestCalculateOrderParameter
# ===========================================================================


class TestCalculateOrderParameter:
    """Unit tests for calculate_order_parameter."""

    def test_empty_angles_returns_zero(self) -> None:
        """Empty input → S = 0.0."""
        result = DAOPCalculator.calculate_order_parameter(np.array([]))
        assert result == pytest.approx(0.0)

    def test_single_angle_returns_one(self) -> None:
        """A single angle is perfectly ordered → S = 1.0."""
        result = DAOPCalculator.calculate_order_parameter(np.array([1.23]))
        assert result == pytest.approx(1.0)

    def test_identical_angles_returns_one(self) -> None:
        """Identical angles at any value → S = 1.0."""
        for angle in [0.0, np.pi / 3, -np.pi / 2, np.pi]:
            angles = np.full(50, angle)
            result = DAOPCalculator.calculate_order_parameter(angles)
            assert result == pytest.approx(
                1.0, abs=1e-10
            ), f"Expected S=1.0 for uniform angle {angle:.3f}, got {result:.6f}"

    def test_opposite_angles_returns_zero(self) -> None:
        """Two angles exactly 180° apart cancel each other → S ≈ 0."""
        # π and -π are equivalent; use well-separated pairs
        angles = np.array([0.0, np.pi])  # sin(0)+sin(π)≈0, cos(0)+cos(π)=0
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_random_uniform_approaches_zero(self) -> None:
        """Uniformly random angles over [0, 2π) → S ≈ 0 for large N."""
        rng = np.random.default_rng(seed=0)
        angles = rng.uniform(0, 2 * np.pi, 50_000)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result < 0.02, f"Expected S<0.02 for 50k random angles, got {result:.4f}"

    def test_result_bounded_0_to_1(self) -> None:
        """S must always be in [0, 1]."""
        rng = np.random.default_rng(seed=7)
        for _ in range(20):
            angles = rng.uniform(-np.pi, np.pi, rng.integers(1, 200))
            result = DAOPCalculator.calculate_order_parameter(angles)
            assert 0.0 <= result <= 1.0 + 1e-12

    def test_well_ordered_threshold(self) -> None:
        """Angles with σ ≲ 24° should give S ≥ 0.9 (Hyberts 1992 convention)."""
        sigma_rad = np.radians(20.0)  # comfortably below ±24° threshold
        angles = _make_ordered_angles(n_models=100, center=-np.pi / 3, sigma=sigma_rad)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result >= 0.9, f"Expected S≥0.9 for σ=20°, got {result:.4f}"

    def test_disordered_threshold(self) -> None:
        """Angles with large spread should give S < 0.9."""
        rng = np.random.default_rng(seed=99)
        angles = rng.uniform(-np.pi, np.pi, 200)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result < 0.5, f"Expected S<0.5 for uniform spread, got {result:.4f}"

    def test_returns_float(self) -> None:
        """Return type must be Python float (not np.float64 subclass)."""
        angles = np.array([1.0, 2.0, 3.0])
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert isinstance(result, float)

    def test_known_formula_value(self) -> None:
        """Verify the Hyberts formula directly for a two-angle case."""
        # Two angles: 0 and π/2
        # sin_sum = sin(0) + sin(π/2) = 0 + 1 = 1
        # cos_sum = cos(0) + cos(π/2) = 1 + 0 = 1
        # S = (1/2) * sqrt(1^2 + 1^2) = (1/2) * sqrt(2)
        angles = np.array([0.0, np.pi / 2])
        expected = 0.5 * np.sqrt(2)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result == pytest.approx(expected, rel=1e-10)


# ===========================================================================
# TestFindWellDefinedResidues
# ===========================================================================


class TestFindWellDefinedResidues:
    """Unit tests for find_well_defined_residues."""

    def test_all_ordered_residues_marked_well_defined(self) -> None:
        """Tightly clustered angles → all residues well-defined."""
        rng = np.random.default_rng(42)
        n_res, n_models = 10, 20
        phi = rng.normal(-np.pi / 3, np.radians(10), (n_res, n_models))
        psi = rng.normal(np.pi / 3, np.radians(10), (n_res, n_models))
        result = DAOPCalculator.find_well_defined_residues(phi, psi)
        assert result.shape == (n_res,)
        assert result.all(), "Expected all residues well-defined for tight angle distribution"

    def test_all_disordered_residues_not_well_defined(self) -> None:
        """Uniformly random angles → no residues well-defined (threshold=1.8)."""
        rng = np.random.default_rng(7)
        n_res, n_models = 10, 200
        phi = rng.uniform(-np.pi, np.pi, (n_res, n_models))
        psi = rng.uniform(-np.pi, np.pi, (n_res, n_models))
        result = DAOPCalculator.find_well_defined_residues(phi, psi)
        assert not result.any(), "Expected no well-defined residues for fully random angles"

    def test_mixed_residues(self) -> None:
        """First half ordered, second half random."""
        rng = np.random.default_rng(0)
        n_ordered, n_disordered, n_models = 5, 5, 200
        n_res = n_ordered + n_disordered

        phi = np.empty((n_res, n_models))
        psi = np.empty((n_res, n_models))

        # Ordered residues
        phi[:n_ordered] = rng.normal(-np.pi / 3, np.radians(8), (n_ordered, n_models))
        psi[:n_ordered] = rng.normal(np.pi / 3, np.radians(8), (n_ordered, n_models))

        # Disordered residues
        phi[n_ordered:] = rng.uniform(-np.pi, np.pi, (n_disordered, n_models))
        psi[n_ordered:] = rng.uniform(-np.pi, np.pi, (n_disordered, n_models))

        result = DAOPCalculator.find_well_defined_residues(phi, psi)
        assert result[:n_ordered].all(), "Ordered residues should all be well-defined"
        assert not result[n_ordered:].any(), "Disordered residues should not be well-defined"

    def test_custom_threshold(self) -> None:
        """Custom threshold changes the cutoff."""
        rng = np.random.default_rng(3)
        n_res, n_models = 5, 50
        # Moderate spread: S≈0.8 each → sum≈1.6; above 1.4 but below 1.8
        phi = rng.normal(-np.pi / 3, np.radians(30), (n_res, n_models))
        psi = rng.normal(np.pi / 3, np.radians(30), (n_res, n_models))

        strict = DAOPCalculator.find_well_defined_residues(phi, psi, threshold=1.8)
        lenient = DAOPCalculator.find_well_defined_residues(phi, psi, threshold=1.4)
        # Lenient threshold should accept at least as many residues
        assert lenient.sum() >= strict.sum()

    def test_output_dtype_is_bool(self) -> None:
        """Return dtype must be boolean."""
        phi = np.full((3, 10), -np.pi / 3)
        psi = np.full((3, 10), np.pi / 3)
        result = DAOPCalculator.find_well_defined_residues(phi, psi)
        assert result.dtype == np.bool_

    def test_single_residue(self) -> None:
        """Should handle single-residue (n_residues=1) ensemble."""
        phi = np.full((1, 20), -np.pi / 3)
        psi = np.full((1, 20), np.pi / 3)
        result = DAOPCalculator.find_well_defined_residues(phi, psi)
        assert result.shape == (1,)
        assert result[0]


# ===========================================================================
# TestCalculateBackboneDaop
# ===========================================================================


class TestCalculateBackboneDaop:
    """Unit tests for calculate_backbone_daop."""

    def test_output_shapes(self) -> None:
        """Each returned array must have shape (n_residues,)."""
        n_res, n_models = 7, 15
        phi = np.random.default_rng(1).normal(0, 0.1, (n_res, n_models))
        psi = np.random.default_rng(2).normal(0, 0.1, (n_res, n_models))
        s_phi, s_psi = DAOPCalculator.calculate_backbone_daop(phi, psi)
        assert s_phi.shape == (n_res,)
        assert s_psi.shape == (n_res,)

    def test_ordered_input_gives_high_s(self) -> None:
        """Tightly clustered angles → S ≥ 0.9 for all residues."""
        n_res, n_models = 5, 30
        phi = np.full((n_res, n_models), -np.pi / 3)
        psi = np.full((n_res, n_models), np.pi / 3)
        s_phi, s_psi = DAOPCalculator.calculate_backbone_daop(phi, psi)
        assert (s_phi >= 0.9).all()
        assert (s_psi >= 0.9).all()

    def test_values_bounded_0_to_1(self) -> None:
        """All S values must be in [0, 1]."""
        rng = np.random.default_rng(99)
        phi = rng.uniform(-np.pi, np.pi, (20, 50))
        psi = rng.uniform(-np.pi, np.pi, (20, 50))
        s_phi, s_psi = DAOPCalculator.calculate_backbone_daop(phi, psi)
        assert (s_phi >= 0).all() and (s_phi <= 1).all()
        assert (s_psi >= 0).all() and (s_psi <= 1).all()

    def test_consistent_with_individual_calls(self) -> None:
        """batch call must match residue-by-residue individual calls."""
        rng = np.random.default_rng(5)
        n_res, n_models = 4, 20
        phi = rng.normal(-np.pi / 3, 0.3, (n_res, n_models))
        psi = rng.normal(np.pi / 3, 0.3, (n_res, n_models))

        s_phi_batch, s_psi_batch = DAOPCalculator.calculate_backbone_daop(phi, psi)

        for i in range(n_res):
            s_phi_i = DAOPCalculator.calculate_order_parameter(phi[i])
            s_psi_i = DAOPCalculator.calculate_order_parameter(psi[i])
            assert s_phi_batch[i] == pytest.approx(s_phi_i, rel=1e-12)
            assert s_psi_batch[i] == pytest.approx(s_psi_i, rel=1e-12)

    def test_independent_phi_psi(self) -> None:
        """φ and ψ order parameters are computed independently."""
        n_res, n_models = 3, 50
        # Ordered phi, random psi
        phi = np.full((n_res, n_models), -np.pi / 3)
        rng = np.random.default_rng(11)
        psi = rng.uniform(-np.pi, np.pi, (n_res, n_models))

        s_phi, s_psi = DAOPCalculator.calculate_backbone_daop(phi, psi)
        assert (s_phi >= 0.99).all(), "Perfectly ordered phi should give S≈1"
        assert (s_psi < 0.5).all(), "Random psi should give low S"


# ===========================================================================
# TestDAOPImports
# ===========================================================================


class TestDAOPImports:
    """Smoke tests verifying the public API is accessible."""

    def test_import_from_submodule(self) -> None:
        """Direct submodule import should work."""
        from synth_pdb.ensemble.daop import DAOPCalculator  # noqa: F401

        assert callable(DAOPCalculator.calculate_order_parameter)

    def test_import_from_ensemble_package(self) -> None:
        """Package-level import should work."""
        from synth_pdb.ensemble import DAOPCalculator  # noqa: F401

        assert callable(DAOPCalculator.calculate_order_parameter)

    def test_all_methods_present(self) -> None:
        """All three public methods must exist."""
        assert hasattr(DAOPCalculator, "calculate_order_parameter")
        assert hasattr(DAOPCalculator, "find_well_defined_residues")
        assert hasattr(DAOPCalculator, "calculate_backbone_daop")


# ===========================================================================
# TestDAOPScientificValidation — Hyberts (1992) formula correctness
# ===========================================================================


class TestDAOPScientificValidation:
    """
    Validate the DAOP formula against known analytical results.

    Reference: Hyberts et al. (1992) Protein Science 1:736-751.
    """

    def test_formula_with_four_evenly_spaced_angles(self) -> None:
        """Four angles at 0, π/2, π, 3π/2 → vector sum cancels → S = 0."""
        angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        # sin_sum = 0+1+0-1 = 0,  cos_sum = 1+0-1+0 = 0
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_formula_with_three_equal_angles(self) -> None:
        """Three identical angles at π/6 → S = 1."""
        angles = np.full(3, np.pi / 6)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_s_is_mean_resultant_length(self) -> None:
        """S is the mean resultant vector length on the unit circle."""
        rng = np.random.default_rng(123)
        angles = rng.uniform(0, 2 * np.pi, 1000)

        # Direct formula
        r = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
        result = DAOPCalculator.calculate_order_parameter(angles)
        assert result == pytest.approx(float(r), rel=1e-9)

    def test_24_degree_std_gives_s_near_0_9(self) -> None:
        """
        A von Mises distribution with σ ≈ 24° should give S ≈ 0.9.

        This is the stated boundary in Hyberts (1992): residues with
        σ ≤ 24° are conventionally considered 'well-ordered'.
        """
        rng = np.random.default_rng(7)
        # Von Mises κ ≈ 1/σ² for small σ; σ=24° ≈ 0.419 rad → κ ≈ 5.7
        kappa = 5.7
        angles = rng.vonmises(0.0, kappa, 10_000)
        result = DAOPCalculator.calculate_order_parameter(angles)
        # Allow ±0.05 tolerance due to finite sample
        assert 0.85 <= result <= 0.95, f"Expected S≈0.9 for κ={kappa} (σ≈24°), got {result:.4f}"

    def test_sum_phi_psi_criterion_matches_pdbstat_convention(self) -> None:
        """
        Verify the PDBStat S(φ)+S(ψ) ≥ 1.8 well-defined criterion is
        equivalent to applying the threshold on the sum of two independent
        order parameters.
        """
        rng = np.random.default_rng(42)
        n_res, n_models = 20, 100
        phi = rng.normal(-np.pi / 3, np.radians(15), (n_res, n_models))
        psi = rng.normal(np.pi / 3, np.radians(15), (n_res, n_models))

        # Manual computation
        s_phi = np.array([DAOPCalculator.calculate_order_parameter(phi[i]) for i in range(n_res)])
        s_psi = np.array([DAOPCalculator.calculate_order_parameter(psi[i]) for i in range(n_res)])
        manual_mask = (s_phi + s_psi) >= 1.8

        # Method result
        method_mask = DAOPCalculator.find_well_defined_residues(phi, psi, threshold=1.8)

        assert np.array_equal(manual_mask, method_mask)
