"""
Tests for synth_pdb.ensemble.statistics — EnsembleStatistics & QualityAssessment.

Coverage target:
    - EnsembleStatistics field defaults and required fields
    - precision property (all three tiers + boundary values)
    - overall_quality property (all four strings)
    - is_single_model / is_ensemble boolean properties
    - to_dict() / from_dict() round-trip fidelity
    - __str__ formatting
    - QualityAssessment to_dict() / from_dict()
    - Public API accessible from synth_pdb.ensemble (import smoke test)
"""

import pytest

from synth_pdb.ensemble.statistics import EnsembleStatistics, QualityAssessment

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def minimal_stats() -> EnsembleStatistics:
    """Minimum required fields only, all optional fields at defaults."""
    return EnsembleStatistics(n_models=5, n_residues=30)


@pytest.fixture()
def high_quality_stats() -> EnsembleStatistics:
    """HIGH precision ensemble — RMSD < 1.0 and well-defined > 80%."""
    return EnsembleStatistics(
        n_models=20,
        n_residues=91,
        mean_pairwise_rmsd=0.85,
        rmsd_to_mean=0.72,
        pct_well_defined=88.0,
        well_defined_residues=80,
    )


@pytest.fixture()
def good_quality_stats() -> EnsembleStatistics:
    """GOOD precision ensemble — 1.0 ≤ RMSD < 2.0."""
    return EnsembleStatistics(
        n_models=10,
        n_residues=60,
        mean_pairwise_rmsd=1.4,
        rmsd_to_mean=1.3,
        pct_well_defined=72.0,
        well_defined_residues=43,
    )


@pytest.fixture()
def moderate_quality_stats() -> EnsembleStatistics:
    """MODERATE precision ensemble — RMSD ≥ 2.0."""
    return EnsembleStatistics(
        n_models=8,
        n_residues=50,
        mean_pairwise_rmsd=2.5,
        rmsd_to_mean=2.3,
        pct_well_defined=45.0,
        well_defined_residues=22,
    )


# ===========================================================================
# TestEnsembleStatisticsFields
# ===========================================================================

class TestEnsembleStatisticsFields:
    """Test field defaults and required-field behaviour."""

    def test_required_fields_stored(self, minimal_stats: EnsembleStatistics) -> None:
        assert minimal_stats.n_models == 5
        assert minimal_stats.n_residues == 30

    def test_optional_field_defaults(self, minimal_stats: EnsembleStatistics) -> None:
        assert minimal_stats.mean_pairwise_rmsd == 0.0
        assert minimal_stats.median_pairwise_rmsd == 0.0
        assert minimal_stats.std_pairwise_rmsd == 0.0
        assert minimal_stats.min_pairwise_rmsd == 0.0
        assert minimal_stats.max_pairwise_rmsd == 0.0
        assert minimal_stats.medoid_index == 0
        assert minimal_stats.medoid_mean_rmsd == 0.0
        assert minimal_stats.rmsd_to_mean == 0.0
        assert minimal_stats.mean_rmsf == 0.0
        assert minimal_stats.max_rmsf == 0.0
        assert minimal_stats.well_defined_residues == 0
        assert minimal_stats.pct_well_defined == 0.0

    def test_full_construction(self) -> None:
        """All fields can be set explicitly."""
        stats = EnsembleStatistics(
            n_models=20,
            n_residues=91,
            mean_pairwise_rmsd=0.85,
            median_pairwise_rmsd=0.80,
            std_pairwise_rmsd=0.12,
            min_pairwise_rmsd=0.35,
            max_pairwise_rmsd=1.20,
            medoid_index=7,
            medoid_mean_rmsd=0.61,
            rmsd_to_mean=0.72,
            mean_rmsf=0.55,
            max_rmsf=1.80,
            well_defined_residues=80,
            pct_well_defined=87.9,
        )
        assert stats.n_models == 20
        assert stats.medoid_index == 7
        assert stats.max_rmsf == pytest.approx(1.80)


# ===========================================================================
# TestEnsembleStatisticsBooleanProperties
# ===========================================================================

class TestEnsembleStatisticsBooleanProperties:
    """Test is_single_model and is_ensemble properties."""

    def test_single_model_true(self) -> None:
        stats = EnsembleStatistics(n_models=1, n_residues=50)
        assert stats.is_single_model is True
        assert stats.is_ensemble is False

    def test_multi_model_ensemble(self) -> None:
        stats = EnsembleStatistics(n_models=20, n_residues=50)
        assert stats.is_single_model is False
        assert stats.is_ensemble is True

    def test_two_models_is_ensemble(self) -> None:
        stats = EnsembleStatistics(n_models=2, n_residues=10)
        assert stats.is_ensemble is True
        assert stats.is_single_model is False


# ===========================================================================
# TestEnsembleStatisticsPrecision
# ===========================================================================

class TestEnsembleStatisticsPrecision:
    """Test precision property across all three tiers and boundary values."""

    def test_high_precision_below_threshold(
        self, high_quality_stats: EnsembleStatistics
    ) -> None:
        assert high_quality_stats.precision == "HIGH"

    def test_good_precision_between_thresholds(
        self, good_quality_stats: EnsembleStatistics
    ) -> None:
        assert good_quality_stats.precision == "GOOD"

    def test_moderate_precision_above_threshold(
        self, moderate_quality_stats: EnsembleStatistics
    ) -> None:
        assert moderate_quality_stats.precision == "MODERATE"

    def test_precision_at_high_boundary(self) -> None:
        """rmsd_to_mean = 0.999… → HIGH; = 1.0 → GOOD."""
        just_below = EnsembleStatistics(n_models=5, n_residues=10, rmsd_to_mean=0.9999)
        at_boundary = EnsembleStatistics(n_models=5, n_residues=10, rmsd_to_mean=1.0)
        assert just_below.precision == "HIGH"
        assert at_boundary.precision == "GOOD"

    def test_precision_at_good_boundary(self) -> None:
        """rmsd_to_mean = 1.999… → GOOD; = 2.0 → MODERATE."""
        just_below = EnsembleStatistics(n_models=5, n_residues=10, rmsd_to_mean=1.9999)
        at_boundary = EnsembleStatistics(n_models=5, n_residues=10, rmsd_to_mean=2.0)
        assert just_below.precision == "GOOD"
        assert at_boundary.precision == "MODERATE"

    def test_exactly_zero_rmsd_is_high(self) -> None:
        stats = EnsembleStatistics(n_models=5, n_residues=10, rmsd_to_mean=0.0)
        assert stats.precision == "HIGH"


# ===========================================================================
# TestEnsembleStatisticsOverallQuality
# ===========================================================================

class TestEnsembleStatisticsOverallQuality:
    """Test overall_quality property covers all four outcome strings."""

    def test_excellent_quality(self, high_quality_stats: EnsembleStatistics) -> None:
        assert high_quality_stats.overall_quality == "Excellent quality NMR ensemble"

    def test_good_quality(self, good_quality_stats: EnsembleStatistics) -> None:
        assert good_quality_stats.overall_quality == "Good quality NMR ensemble"

    def test_acceptable_quality_moderate_precision(
        self, moderate_quality_stats: EnsembleStatistics
    ) -> None:
        assert moderate_quality_stats.overall_quality == "Acceptable quality NMR ensemble"

    def test_may_need_refinement(self) -> None:
        """HIGH precision but low well-defined % → may need refinement."""
        stats = EnsembleStatistics(
            n_models=10,
            n_residues=100,
            rmsd_to_mean=0.5,   # HIGH precision
            pct_well_defined=40.0,  # < 80% → not "Excellent"
        )
        # precision="HIGH" but pct_well_defined <= 60 → "may need refinement"
        assert stats.overall_quality == "Ensemble may need refinement"

    def test_acceptable_via_well_defined_pct(self) -> None:
        """GOOD precision but pct_well_defined > 60 → Acceptable."""
        stats = EnsembleStatistics(
            n_models=10,
            n_residues=100,
            rmsd_to_mean=2.5,   # MODERATE precision
            pct_well_defined=65.0,  # > 60 → Acceptable
        )
        assert stats.overall_quality == "Acceptable quality NMR ensemble"


# ===========================================================================
# TestEnsembleStatisticsSerialisation
# ===========================================================================

class TestEnsembleStatisticsSerialisation:
    """Test to_dict() and from_dict() round-trips."""

    def test_to_dict_contains_all_fields(
        self, high_quality_stats: EnsembleStatistics
    ) -> None:
        d = high_quality_stats.to_dict()
        expected_keys = {
            "n_models", "n_residues",
            "mean_pairwise_rmsd", "median_pairwise_rmsd",
            "std_pairwise_rmsd", "min_pairwise_rmsd", "max_pairwise_rmsd",
            "medoid_index", "medoid_mean_rmsd",
            "rmsd_to_mean",
            "mean_rmsf", "max_rmsf",
            "well_defined_residues", "pct_well_defined",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_fields(
        self, high_quality_stats: EnsembleStatistics
    ) -> None:
        d = high_quality_stats.to_dict()
        assert d["n_models"] == 20
        assert d["rmsd_to_mean"] == pytest.approx(0.72)
        assert d["pct_well_defined"] == pytest.approx(88.0)

    def test_from_dict_round_trip_equality(
        self, high_quality_stats: EnsembleStatistics
    ) -> None:
        reconstructed = EnsembleStatistics.from_dict(high_quality_stats.to_dict())
        assert reconstructed == high_quality_stats

    def test_from_dict_with_minimal_keys(self) -> None:
        """from_dict must work with only required fields present."""
        d: dict = {"n_models": 3, "n_residues": 20}
        stats = EnsembleStatistics.from_dict(d)
        assert stats.n_models == 3
        assert stats.rmsd_to_mean == 0.0  # default

    def test_from_dict_ignores_extra_keys(self) -> None:
        """Extra keys in the dict must be silently ignored."""
        d = {
            "n_models": 5,
            "n_residues": 10,
            "unknown_future_field": 99.9,
        }
        stats = EnsembleStatistics.from_dict(d)
        assert stats.n_models == 5

    def test_from_dict_raises_on_missing_required_key(self) -> None:
        """Missing required key → KeyError."""
        with pytest.raises(KeyError):
            EnsembleStatistics.from_dict({"n_models": 5})  # n_residues missing

    def test_round_trip_preserves_floats(self) -> None:
        """Floating-point values survive serialisation unchanged."""
        stats = EnsembleStatistics(
            n_models=20,
            n_residues=91,
            rmsd_to_mean=0.123456789,
            pct_well_defined=78.654321,
        )
        reconstructed = EnsembleStatistics.from_dict(stats.to_dict())
        assert reconstructed.rmsd_to_mean == pytest.approx(stats.rmsd_to_mean, rel=1e-10)
        assert reconstructed.pct_well_defined == pytest.approx(stats.pct_well_defined, rel=1e-10)


# ===========================================================================
# TestEnsembleStatisticsStr
# ===========================================================================

class TestEnsembleStatisticsStr:
    """Test __str__ output contains expected content."""

    def test_str_contains_n_models(self, high_quality_stats: EnsembleStatistics) -> None:
        s = str(high_quality_stats)
        assert "20" in s

    def test_str_contains_n_residues(self, high_quality_stats: EnsembleStatistics) -> None:
        s = str(high_quality_stats)
        assert "91" in s

    def test_str_contains_precision(self, high_quality_stats: EnsembleStatistics) -> None:
        s = str(high_quality_stats)
        assert "HIGH" in s

    def test_str_contains_quality(self, high_quality_stats: EnsembleStatistics) -> None:
        s = str(high_quality_stats)
        assert "Excellent" in s

    def test_str_is_multiline(self, minimal_stats: EnsembleStatistics) -> None:
        s = str(minimal_stats)
        assert "\n" in s


# ===========================================================================
# TestQualityAssessment
# ===========================================================================

class TestQualityAssessment:
    """Tests for QualityAssessment dataclass."""

    def test_fields_stored(self) -> None:
        qa = QualityAssessment(precision="HIGH", overall="Excellent quality NMR ensemble")
        assert qa.precision == "HIGH"
        assert qa.overall == "Excellent quality NMR ensemble"

    def test_to_dict_keys(self) -> None:
        qa = QualityAssessment(precision="GOOD", overall="Good quality NMR ensemble")
        d = qa.to_dict()
        assert set(d.keys()) == {"precision", "overall"}

    def test_to_dict_values(self) -> None:
        qa = QualityAssessment(precision="MODERATE", overall="Acceptable quality NMR ensemble")
        d = qa.to_dict()
        assert d["precision"] == "MODERATE"
        assert d["overall"] == "Acceptable quality NMR ensemble"

    def test_from_dict_round_trip(self) -> None:
        qa = QualityAssessment(precision="HIGH", overall="Excellent quality NMR ensemble")
        reconstructed = QualityAssessment.from_dict(qa.to_dict())
        assert reconstructed == qa

    def test_from_dict_raises_on_missing_key(self) -> None:
        with pytest.raises(KeyError):
            QualityAssessment.from_dict({"precision": "HIGH"})  # "overall" missing


# ===========================================================================
# TestStatisticsImports
# ===========================================================================

class TestStatisticsImports:
    """Smoke tests verifying the public API is accessible."""

    def test_import_from_submodule(self) -> None:
        from synth_pdb.ensemble.statistics import (  # noqa: F401
            EnsembleStatistics,
            QualityAssessment,
        )
        assert EnsembleStatistics is not None
        assert QualityAssessment is not None

    def test_import_from_ensemble_package(self) -> None:
        from synth_pdb.ensemble import (  # noqa: F401
            EnsembleStatistics,
            QualityAssessment,
        )
        assert EnsembleStatistics is not None
        assert QualityAssessment is not None

    def test_ensemble_statistics_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(EnsembleStatistics)

    def test_quality_assessment_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(QualityAssessment)
