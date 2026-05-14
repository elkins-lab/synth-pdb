"""Tests for synth_pdb/quality/gnn/gnn_classifier.py - targeting uncovered lines:
- Lines 185-186: predict() ImportError when torch is absent
- Lines 240-241: save() path (writes a checkpoint to disk)
- Lines 280-281: load() ImportError when torch is absent
- Lines 303-305: load() exception path.
"""

import os
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed; skipping GNN classifier tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fresh_classifier():
    """Return a GNNQualityClassifier with a randomly-initialized model (no checkpoint)."""
    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

    return GNNQualityClassifier()  # no checkpoint -> uses _init_fresh_model()


def _make_helix_pdb(length: int = 12) -> str:
    from synth_pdb.generator import generate_pdb_content

    return generate_pdb_content(length=length, conformation="alpha", minimize_energy=False)


# ---------------------------------------------------------------------------
# predict() - torch ImportError (lines 185-186)
# ---------------------------------------------------------------------------


class TestGNNPredictImportError:
    def test_predict_raises_importerror_without_torch(self, tmp_path):
        """When torch cannot be imported inside predict(), an ImportError with a
        helpful message should be raised.
        """
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        clf = GNNQualityClassifier.__new__(GNNQualityClassifier)
        clf.model = None  # skip constructor side-effects

        pdb_str = _make_helix_pdb()

        # Hide torch inside the gnn_classifier module scope
        with patch.dict(
            "sys.modules", {"torch": None, "torch_geometric": None, "torch_geometric.data": None}
        ):
            with pytest.raises(ImportError, match="torch"):
                clf.predict(pdb_str)


# ---------------------------------------------------------------------------
# save() / load() round-trip (lines 240-241, 280-305)
# ---------------------------------------------------------------------------


class TestGNNSaveLoad:
    def test_save_creates_checkpoint_file(self, tmp_path):
        """save() must write a .pt file to the specified path (lines 240-241)."""
        clf = _make_fresh_classifier()
        ckpt_path = str(tmp_path / "test_gnn.pt")

        clf.save(ckpt_path)

        assert os.path.exists(ckpt_path), f"Checkpoint not written to {ckpt_path}"
        assert os.path.getsize(ckpt_path) > 0

    def test_save_sets_model_path_attribute(self, tmp_path):
        """After save(), _model_path should point to the checkpoint."""
        clf = _make_fresh_classifier()
        ckpt_path = str(tmp_path / "gnn_v1.pt")
        clf.save(ckpt_path)
        assert clf._model_path == ckpt_path

    def test_load_restores_model(self, tmp_path):
        """load() must reconstruct the model from the checkpoint (lines 285-302)."""
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        # Save a fresh model
        clf_a = _make_fresh_classifier()
        ckpt_path = str(tmp_path / "roundtrip.pt")
        clf_a.save(ckpt_path)

        # Load it into a second classifier
        clf_b = GNNQualityClassifier(model_path=ckpt_path)
        assert clf_b.model is not None
        assert clf_b._model_path == ckpt_path

    def test_load_exception_path(self, tmp_path):
        """load() must re-raise if the checkpoint is corrupt (lines 303-305)."""
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        bad_path = str(tmp_path / "corrupt.pt")
        with open(bad_path, "w") as f:
            f.write("this is not a real pytorch checkpoint")

        with pytest.raises(Exception):
            GNNQualityClassifier(model_path=bad_path)

    def test_load_importerror_without_torch(self, tmp_path):
        """load() must raise ImportError with a helpful message when torch is absent
        (lines 280-281).
        """
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        clf = GNNQualityClassifier.__new__(GNNQualityClassifier)
        clf.model = None

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="torch"):
                clf.load("some/path.pt")


# ---------------------------------------------------------------------------
# predict() - happy path with fresh (untrained) model
# ---------------------------------------------------------------------------


class TestGNNPredictHappyPath:
    def test_predict_returns_correct_types(self):
        """Fresh model must return (bool, float, dict) from predict()."""
        clf = _make_fresh_classifier()
        pdb_str = _make_helix_pdb(15)

        is_good, prob, feat_dict = clf.predict(pdb_str)

        assert isinstance(is_good, bool)
        assert isinstance(prob, float)
        assert isinstance(feat_dict, dict)
        assert 0.0 <= prob <= 1.0

    def test_predict_feature_dict_has_expected_keys(self):
        """Feature dict must contain exactly the 8 expected feature names."""
        from synth_pdb.quality.gnn.gnn_classifier import _FEATURE_NAMES

        clf = _make_fresh_classifier()
        pdb_str = _make_helix_pdb(10)

        _, _, feat_dict = clf.predict(pdb_str)

        assert set(feat_dict.keys()) == set(_FEATURE_NAMES), (
            f"Feature dict keys don't match _FEATURE_NAMES.\n"
            f"Got:      {set(feat_dict.keys())}\n"
            f"Expected: {set(_FEATURE_NAMES)}"
        )

    def test_predict_probability_is_bounded(self):
        """P(Good) must be strictly in [0, 1]."""
        clf = _make_fresh_classifier()
        for length in [8, 20, 50]:
            pdb_str = _make_helix_pdb(length)
            _, prob, _ = clf.predict(pdb_str)
            assert 0.0 <= prob <= 1.0, f"prob={prob} out of bounds for length={length}"

    def test_save_importerror_without_torch(self, tmp_path):
        """save() must raise ImportError when torch is absent."""
        clf = _make_fresh_classifier()
        ckpt_path = str(tmp_path / "will_fail.pt")

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="torch"):
                clf.save(ckpt_path)

    def test_init_missing_default_model_logs_info(self, caplog):
        """Initialization without a model_path should log an info message and
        create a random-weight model when the default checkpoint is missing
        (lines 136-146).
        """
        import logging

        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        with caplog.at_level(logging.INFO):
            with patch("os.path.exists", return_value=False):
                # Ensure we also mock the internal init so we don't accidentally do full setup
                with patch.object(GNNQualityClassifier, "_init_fresh_model") as mock_init:
                    GNNQualityClassifier(model_path=None)

                    assert "No pre-trained GNN checkpoint found" in caplog.text
                    assert "random weights" in caplog.text
                    mock_init.assert_called_once()
