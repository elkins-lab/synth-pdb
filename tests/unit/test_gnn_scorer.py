"""tests/unit/test_gnn_scorer.py.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the GNN quality scorer public API.

Tests cover:
  - score_structure() on known-good (helix) and known-bad (random) structures
  - Per-residue pLDDT length and value range
  - score_batch() consistency with single-structure scoring
  - QualityScore dataclass fields
  - File-path input to score_structure()
  - GNNQualityClassifier import from synth_pdb.quality
"""

import os
import tempfile

import pytest

# Mark entire module as requiring torch + torch_geometric
pytestmark = pytest.mark.gnn


@pytest.fixture(scope="module")
def helix_pdb():
    """A 20-residue alpha-helix - should score as High Quality."""
    from synth_pdb.generator import generate_pdb_content

    return generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)


@pytest.fixture(scope="module")
def random_pdb():
    """A 20-residue random coil - should score as Low Quality."""
    from synth_pdb.generator import generate_pdb_content

    return generate_pdb_content(length=20, conformation="random", minimize_energy=False)


# -----------------------------------------------------------------------------
# score_structure() - single-structure API
# -----------------------------------------------------------------------------


class TestScoreStructure:
    def test_helix_scores_high_quality(self, helix_pdb):
        from synth_pdb.score import _get_classifier, score_structure

        if not _get_classifier().is_pretrained:
            pytest.skip("No pre-trained model found - skipping accuracy assertion")

        result = score_structure(helix_pdb)
        assert result.label == "High Quality", (
            f"Expected helix to be High Quality but got {result.label} "
            f"(score={result.global_score:.4f})"
        )
        assert result.global_score > 0.5

    def test_random_coil_scores_low_quality(self, random_pdb):
        from synth_pdb.score import _get_classifier, score_structure

        if not _get_classifier().is_pretrained:
            pytest.skip("No pre-trained model found - skipping accuracy assertion")

        result = score_structure(random_pdb)
        assert result.label == "Low Quality", (
            f"Expected random coil to be Low Quality but got {result.label} "
            f"(score={result.global_score:.4f})"
        )
        assert result.global_score < 0.5

    def test_global_score_in_unit_interval(self, helix_pdb, random_pdb):
        from synth_pdb.score import score_structure

        for pdb in (helix_pdb, random_pdb):
            result = score_structure(pdb)
            assert (
                0.0 <= result.global_score <= 1.0
            ), f"global_score={result.global_score} is outside [0, 1]"

    def test_helix_scores_higher_than_random(self, helix_pdb, random_pdb):
        from synth_pdb.score import _get_classifier, score_structure

        if not _get_classifier().is_pretrained:
            pytest.skip("No pre-trained model found - skipping accuracy assertion")

        helix_result = score_structure(helix_pdb)
        random_result = score_structure(random_pdb)
        assert (
            helix_result.global_score > random_result.global_score
        ), "Helix should score higher than random coil"

    def test_file_path_input(self, helix_pdb):
        """score_structure() should accept a file path as well as a PDB string."""
        from synth_pdb.score import _get_classifier, score_structure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(helix_pdb)
            tmp_path = f.name
        try:
            result = score_structure(tmp_path)
            if _get_classifier().is_pretrained:
                assert result.global_score > 0.5
        finally:
            os.unlink(tmp_path)

    def test_missing_file_raises_file_not_found(self):
        from synth_pdb.score import score_structure

        with pytest.raises(FileNotFoundError):
            score_structure("/nonexistent/path/protein.pdb")


# -----------------------------------------------------------------------------
# Per-residue pLDDT output
# -----------------------------------------------------------------------------


class TestPerResiduePLDDT:
    def test_per_residue_length_matches_n_residues(self, helix_pdb):
        from synth_pdb.score import score_structure

        result = score_structure(helix_pdb)
        assert (
            len(result.per_residue) == result.n_residues
        ), f"per_residue length {len(result.per_residue)} != n_residues {result.n_residues}"

    def test_per_residue_labels_length_matches(self, helix_pdb):
        from synth_pdb.score import score_structure

        result = score_structure(helix_pdb)
        assert len(result.residue_labels) == len(result.per_residue)

    def test_per_residue_values_in_unit_interval(self, helix_pdb):
        from synth_pdb.score import score_structure

        result = score_structure(helix_pdb)
        for i, score in enumerate(result.per_residue):
            assert 0.0 <= score <= 1.0, f"per_residue[{i}]={score} outside [0, 1]"

    def test_residue_labels_valid_values(self, helix_pdb):
        from synth_pdb.score import score_structure

        valid = {"Very High", "High", "Uncertain", "Low"}
        result = score_structure(helix_pdb)
        for i, label in enumerate(result.residue_labels):
            assert label in valid, f"residue_labels[{i}]={label!r} is not a valid pLDDT band"

    def test_helix_residues_mostly_very_high(self, helix_pdb):
        """For a clean alpha helix, most residues should be Very High confidence."""
        from synth_pdb.score import _get_classifier, score_structure

        if not _get_classifier().is_pretrained:
            pytest.skip("No pre-trained model found - skipping accuracy assertion")

        result = score_structure(helix_pdb)
        very_high_count = result.residue_labels.count("Very High")
        fraction = very_high_count / len(result.residue_labels)
        assert fraction > 0.5, (
            f"Expected >50% Very High for a helix, got {fraction:.0%} "
            f"({very_high_count}/{len(result.residue_labels)})"
        )


# -----------------------------------------------------------------------------
# score_batch() - batch API
# -----------------------------------------------------------------------------


class TestScoreBatch:
    def test_batch_returns_correct_length(self, helix_pdb, random_pdb):
        from synth_pdb.score import score_batch

        results = score_batch([helix_pdb, random_pdb])
        assert len(results) == 2

    def test_batch_consistent_with_single(self, helix_pdb, random_pdb):
        """Batch scores should match single-call scores (same model, same input)."""
        from synth_pdb.score import score_batch, score_structure

        batch_results = score_batch([helix_pdb, random_pdb])
        single_helix = score_structure(helix_pdb)
        single_random = score_structure(random_pdb)

        assert abs(batch_results[0].global_score - single_helix.global_score) < 1e-5
        assert abs(batch_results[1].global_score - single_random.global_score) < 1e-5

    def test_empty_batch_returns_empty_list(self):
        from synth_pdb.score import score_batch

        assert score_batch([]) == []


# -----------------------------------------------------------------------------
# QualityScore dataclass
# -----------------------------------------------------------------------------


class TestQualityScoreDataclass:
    def test_quality_score_has_expected_fields(self, helix_pdb):
        from synth_pdb.score import score_structure

        result = score_structure(helix_pdb)
        assert hasattr(result, "global_score")
        assert hasattr(result, "label")
        assert hasattr(result, "per_residue")
        assert hasattr(result, "residue_labels")
        assert hasattr(result, "features")
        assert hasattr(result, "n_residues")

    def test_features_dict_has_expected_keys(self, helix_pdb):
        from synth_pdb.score import score_structure

        result = score_structure(helix_pdb)
        expected_keys = {
            "sin_phi",
            "cos_phi",
            "sin_psi",
            "cos_psi",
            "b_factor_norm",
            "seq_position",
            "is_n_terminus",
            "is_c_terminus",
        }
        assert set(result.features.keys()) == expected_keys


# -----------------------------------------------------------------------------
# Import convenience
# -----------------------------------------------------------------------------


class TestImports:
    def test_gnn_quality_classifier_importable_from_quality(self):
        from synth_pdb.quality import GNNQualityClassifier, QualityScore  # noqa: F401

        assert GNNQualityClassifier is not None

    def test_score_structure_importable_from_score(self):
        from synth_pdb.score import score_structure  # noqa: F401

        assert callable(score_structure)

    def test_quality_score_importable_from_score(self):
        from synth_pdb.score import QualityScore  # noqa: F401

        assert QualityScore is not None
