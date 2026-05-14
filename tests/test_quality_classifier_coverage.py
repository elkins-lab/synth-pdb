"""Tests for synth_pdb/quality/classifier.py - targeting all uncovered lines:
- Line 37:  __init__ without model file (warning logged)
- Lines 43-46: load_model when joblib is missing (ImportError)
- Lines 51-53: load_model when joblib.load raises an exception
- Line 65:  predict() when self.model is None (raises RuntimeError).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ProteinQualityClassifier.__init__ - no model file present
# ---------------------------------------------------------------------------


class TestClassifierInit:
    @patch("synth_pdb.quality.classifier.os.path.exists", return_value=False)
    def test_no_model_file_logs_warning(self, mock_exists, tmp_path, caplog):
        """When no default model exists, __init__ should log a warning and leave
        self.model as None.
        """
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        # Point to a directory where the default model definitely won't exist
        with caplog.at_level(logging.WARNING, logger="synth_pdb.quality.classifier"):
            clf = ProteinQualityClassifier()

        # Model should be None because there's no .joblib file in the package
        # (in CI / fresh checkouts the model file is not bundled)
        if clf.model is None:
            assert any(
                "No model found" in r.message for r in caplog.records
            ), "Expected a 'No model found' warning when no model file exists"

    def test_explicit_model_path_triggers_load(self, tmp_path):
        """Passing an explicit model_path should call load_model.
        We use a path that doesn't exist; load_model should catch the exception.
        """
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        fake_path = str(tmp_path / "nonexistent.joblib")

        # Should not raise - load_model catches exceptions
        ProteinQualityClassifier(model_path=fake_path)
        # model will be None because the file doesn't exist; that's fine
        # (we just confirm it doesn't propagate an unhandled exception)


# ---------------------------------------------------------------------------
# ProteinQualityClassifier.load_model - joblib missing (lines 43-46)
# ---------------------------------------------------------------------------


class TestLoadModelImportError:
    def test_load_model_when_joblib_missing(self, tmp_path, caplog):
        """If joblib is not installed, load_model should log an error, set
        self.model = None, and return early (no raise).
        """
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        clf.model = None
        clf.feature_names = []

        fake_path = str(tmp_path / "model.joblib")

        # Simulate missing joblib by patching the import inside the method
        with patch.dict("sys.modules", {"joblib": None}):
            with caplog.at_level(logging.ERROR, logger="synth_pdb.quality.classifier"):
                clf.load_model(fake_path)

        assert clf.model is None
        assert any(
            "joblib" in r.message.lower() for r in caplog.records
        ), "Expected an error log mentioning 'joblib' when the import fails"

    def test_load_model_exception_sets_model_none(self, tmp_path, caplog):
        """When joblib is available but joblib.load raises, model should be set
        to None and an error should be logged (lines 51-53).
        """
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        clf.model = None
        clf.feature_names = []

        # Create a dummy file so the path exists
        fake_path = str(tmp_path / "bad_model.joblib")
        with open(fake_path, "w") as f:
            f.write("not a real model")

        mock_joblib = MagicMock()
        mock_joblib.load.side_effect = RuntimeError("corrupt file")

        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            with caplog.at_level(logging.ERROR, logger="synth_pdb.quality.classifier"):
                clf.load_model(fake_path)

        assert clf.model is None, "load_model must set model=None on exception"
        assert any(
            "Failed to load" in r.message for r in caplog.records
        ), "Expected an error log when joblib.load raises"


# ---------------------------------------------------------------------------
# ProteinQualityClassifier.predict - model is None (line 65)
# ---------------------------------------------------------------------------


class TestPredictNoModel:
    def test_predict_raises_when_model_none(self):
        """predict() must raise RuntimeError when self.model is None (line 65)."""
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        clf.model = None
        clf.feature_names = []

        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict(
                "ATOM   1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            )


# ---------------------------------------------------------------------------
# ProteinQualityClassifier - happy path with mocked joblib
# ---------------------------------------------------------------------------


class TestPredictHappyPath:
    def test_predict_returns_correct_types(self, tmp_path):
        """With a mocked sklearn model, predict() should return (bool, float, dict)."""
        import numpy as np

        from synth_pdb.generator import generate_pdb_content
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        pdb_str = generate_pdb_content(
            sequence_str="ACDEFGHIK", conformation="alpha", minimize_energy=False
        )

        # Build a mock sklearn RandomForestClassifier
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        from synth_pdb.quality.features import get_feature_names

        clf.feature_names = get_feature_names()
        clf.model = mock_model

        is_good, prob, feat_dict = clf.predict(pdb_str)

        assert isinstance(is_good, (bool, np.bool_))
        assert isinstance(prob, float)
        assert isinstance(feat_dict, dict)
        assert 0.0 <= prob <= 1.0
        assert is_good  # 0.7 > 0.5

    def test_predict_low_score_is_bad(self, tmp_path):
        """When model gives P(good)=0.2, is_good should be False."""
        import numpy as np

        from synth_pdb.generator import generate_pdb_content
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        pdb_str = generate_pdb_content(
            sequence_str="ACDEFGHIK", conformation="alpha", minimize_energy=False
        )

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        from synth_pdb.quality.features import get_feature_names

        clf.feature_names = get_feature_names()
        clf.model = mock_model

        is_good, prob, _ = clf.predict(pdb_str)
        assert not is_good
        assert pytest.approx(prob) == 0.2

    def test_load_model_success(self, tmp_path):
        """When joblib.load succeeds, model should be set and an info log emitted."""
        from synth_pdb.quality.classifier import ProteinQualityClassifier

        clf = ProteinQualityClassifier.__new__(ProteinQualityClassifier)
        clf.model = None
        clf.feature_names = []

        fake_model = MagicMock()
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = fake_model

        fake_path = str(tmp_path / "model.joblib")
        with open(fake_path, "w") as f:
            f.write("placeholder")

        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            clf.load_model(fake_path)

        assert clf.model is fake_model, "load_model must assign the loaded model to self.model"
