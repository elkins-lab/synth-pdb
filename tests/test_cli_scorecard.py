from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_pdb.main import main


class TestCLIScorecard:
    """Test suite for the Integrated Scientific Defense Scorecard CLI integration."""

    def test_scorecard_basic(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify that --scorecard prints the unified table."""
        output_file = tmp_path / "test.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAA",
            "--scorecard",
            "--output",
            str(output_file),
        ]

        with patch("sys.argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "INTEGRATED SCIENTIFIC DEFENSE SCORECARD" in captured.out
        assert "PHYSICS & GEOMETRY" in captured.out
        assert "BIOPHYSICAL REALISM" in captured.out
        assert "OVERALL:" in captured.out

    def test_scorecard_with_ml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify that --include-ml adds the AI layer to the scorecard."""
        output_file = tmp_path / "test_ml.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAA",
            "--scorecard",
            "--include-ml",
            "--output",
            str(output_file),
        ]

        # Mock classifier to avoid loading model file
        with patch("synth_pdb.quality.classifier.ProteinQualityClassifier") as mock_clf_class:
            mock_clf = mock_clf_class.return_value
            mock_clf.model = MagicMock()
            mock_clf.predict.return_value = (True, 0.99, {})

            with patch("sys.argv", test_args):
                main()

        captured = capsys.readouterr()
        assert "AI/GNN QUALITY FILTER" in captured.out
        # Flexible check for value
        assert "0.99" in captured.out
        assert "ML Confidence Score" in captured.out

    def test_scorecard_with_bmrb(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify that --bmrb-id adds the NMR layer to the scorecard."""
        output_file = tmp_path / "test_bmrb.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAA",
            "--scorecard",
            "--bmrb-id",
            "6457",
            "--output",
            str(output_file),
        ]

        # Mock BMRB fetching and PDBValidator method to ensure satisfaction
        mock_restraints = [
            {
                "index_1": 1,
                "atom_name_1": "N",
                "index_2": 1,
                "atom_name_2": "CA",
                "upper_limit": 5.0,
            }
        ]

        with patch("synth_pdb.bmrb_api.BMRBAPI.fetch_restraints", return_value=mock_restraints):
            with patch("sys.argv", test_args):
                main()

        captured = capsys.readouterr()
        assert "NMR SPECTROSCOPIC FIDELITY" in captured.out
        assert "NOE Satisfaction" in captured.out
        assert "100.0%" in captured.out

    def test_scorecard_with_invalid_bmrb(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Verify scorecard handles BMRB fetch failure gracefully."""
        output_file = tmp_path / "test_bad_bmrb.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAA",
            "--scorecard",
            "--bmrb-id",
            "invalid_id",
            "--output",
            str(output_file),
        ]

        # Mock BMRB failure
        with patch("synth_pdb.bmrb_api.BMRBAPI.fetch_restraints", return_value=[]):
            with patch("sys.argv", test_args):
                main()

        captured = capsys.readouterr()
        # NMR layer should be missing if no restraints fetched
        assert "NMR SPECTROSCOPIC FIDELITY" not in captured.out
        assert "OVERALL:" in captured.out

    def test_scorecard_interface_metrics_single_chain(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Verify scorecard does not show interface layer for single chain."""
        output_file = tmp_path / "test_single.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AAA",
            "--scorecard",
            "--output",
            str(output_file),
        ]

        with patch("sys.argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "STRUCTURAL INTERACTOME" not in captured.out
