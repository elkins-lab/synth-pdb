import os
import sys
import tempfile
from unittest.mock import patch

from synth_pdb import main


def test_main_rpf_validation_cli() -> None:
    """Verify RPF validation path in main CLI."""
    # Create a mock restraint file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".upl", delete=False) as f:
        f.write("1 N 1 H 3.0\n2 N 2 H 3.5\n")
        upl_path = f.name

    test_args = [
        "synth-pdb",
        "--sequence",
        "AA",
        "--restraints",
        upl_path,
        "--output",
        "test_rpf.pdb",
    ]

    try:
        with patch.object(sys, "argv", test_args):
            with patch("synth_pdb.nmr.calculate_rpf_score") as mock_rpf:
                mock_rpf.return_value = {"recall": 1.0, "precision": 1.0, "f_measure": 1.0}
                main.main()
                assert mock_rpf.called
    finally:
        if os.path.exists(upl_path):
            os.remove(upl_path)
        if os.path.exists("test_rpf.pdb"):
            os.remove("test_rpf.pdb")


def test_main_rdc_validation_cli() -> None:
    """Verify RDC validation path in main CLI."""
    # Create a mock RDC file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rdc", delete=False) as f:
        f.write("1 N 1 H 10.0\n2 N 2 H -5.0\n")
        rdc_path = f.name

    test_args = [
        "synth-pdb",
        "--sequence",
        "AA",
        "--rdc-restraints",
        rdc_path,
        "--rdc-da",
        "10.0",
        "--output",
        "test_rdc.pdb",
    ]

    try:
        with patch.object(sys, "argv", test_args):
            with patch("synth_pdb.rdc.calculate_rdcs") as mock_calc:
                # Mock return {res_id: value}
                mock_calc.return_value = {1: 10.0, 2: -5.0}
                main.main()
                assert mock_calc.called
    finally:
        if os.path.exists(rdc_path):
            os.remove(rdc_path)
        if os.path.exists("test_rdc.pdb"):
            os.remove("test_rdc.pdb")


def test_main_quality_filter_cli() -> None:
    """Verify Quality Filter path in main CLI."""
    test_args = [
        "synth-pdb",
        "--sequence",
        "AAA",
        "--quality-filter",
        "--quality-score-cutoff",
        "0.5",
        "--output",
        "test_q.pdb",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("synth_pdb.quality.classifier.ProteinQualityClassifier.predict") as mock_predict:
            # Mock pass on first attempt
            mock_predict.return_value = (True, 0.8, {})
            main.main()
            assert mock_predict.called

    if os.path.exists("test_q.pdb"):
        os.remove("test_q.pdb")
