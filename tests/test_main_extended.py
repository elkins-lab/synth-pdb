from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from synth_pdb.main import main


def test_main_cryo_em_happy_path() -> None:
    """Test cryo-em mode successful execution path."""
    with (
        patch(
            "sys.argv", ["synth-pdb", "--mode", "cryo-em", "--sequence", "AC", "--n-decoys", "2"]
        ),
        patch("synth_pdb.batch_generator.BatchedGenerator") as mock_bg_class,
        patch(
            "synth_pdb.cryo_em.generate_density_map",
            return_value=(np.zeros((10, 10, 10)), [0, 0, 0]),
        ),
        patch("synth_pdb.cryo_em.save_mrc_file") as mock_save,
    ):
        # Setup mock batch and stack
        mock_batch = MagicMock()
        mock_stack = MagicMock()
        mock_batch.to_stack.return_value = mock_stack
        mock_bg_instance = mock_bg_class.return_value
        mock_bg_instance.generate_batch.return_value = mock_batch

        main()

        mock_bg_instance.generate_batch.assert_called_once()
        mock_save.assert_called_once()


def test_main_cryo_em_missing_sequence() -> None:
    """Test cryo-em mode failure when sequence and length are missing."""
    with (
        patch(
            "sys.argv", ["synth-pdb", "--mode", "cryo-em", "--length", "0", "--log-level", "ERROR"]
        ),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(1)


def test_main_cryo_em_seed_propagation() -> None:
    """Verify --seed is passed to sequence generation in cryo-em mode."""
    with (
        patch("sys.argv", ["synth-pdb", "--mode", "cryo-em", "--length", "5", "--seed", "42"]),
        patch("synth_pdb.generator._get_random_sequence", return_value=["ALA"]) as mock_get_seq,
        patch("synth_pdb.batch_generator.BatchedGenerator"),
        patch(
            "synth_pdb.cryo_em.generate_density_map", return_value=(np.zeros((1, 1, 1)), [0, 0, 0])
        ),
        patch("synth_pdb.cryo_em.save_mrc_file"),
    ):
        main()
        assert mock_get_seq.called
        _, kwargs = mock_get_seq.call_args
        assert "rng" in kwargs
        assert kwargs["rng"] is not None


def test_main_saxs_seed_propagation() -> None:
    """Verify --seed is passed to sequence generation in saxs mode."""
    with (
        patch("sys.argv", ["synth-pdb", "--mode", "saxs", "--length", "5", "--seed", "42"]),
        patch("synth_pdb.generator._get_random_sequence", return_value=["ALA"]) as mock_get_seq,
        patch("synth_pdb.batch_generator.BatchedGenerator") as mock_bg_class,
        patch("synth_pdb.saxs.calculate_saxs_profile", return_value=(np.zeros(1), np.zeros(1))),
        patch("synth_pdb.saxs.export_saxs_profile"),
    ):
        mock_batch = MagicMock()
        mock_stack = [MagicMock()]
        mock_batch.to_stack.return_value = mock_stack
        mock_bg_class.return_value.generate_batch.return_value = mock_batch

        main()
        assert mock_get_seq.called
        _, kwargs = mock_get_seq.call_args
        assert "rng" in kwargs
        assert kwargs["rng"] is not None


def test_main_saxs_happy_path() -> None:
    """Test saxs mode successful execution path."""
    with (
        patch("sys.argv", ["synth-pdb", "--mode", "saxs", "--length", "5", "--n-decoys", "2"]),
        patch("synth_pdb.batch_generator.BatchedGenerator") as mock_bg_class,
        patch("synth_pdb.saxs.calculate_saxs_profile", return_value=(np.zeros(51), np.zeros(51))),
        patch("synth_pdb.saxs.export_saxs_profile") as mock_export,
    ):
        mock_batch = MagicMock()
        mock_stack = [MagicMock(), MagicMock()]  # simulate 2 decoys
        mock_batch.to_stack.return_value = mock_stack
        mock_bg_instance = mock_bg_class.return_value
        mock_bg_instance.generate_batch.return_value = mock_batch

        main()

        assert mock_export.called


def test_main_saxs_missing_sequence() -> None:
    """Test saxs mode failure when sequence and length are missing."""
    with (
        patch("sys.argv", ["synth-pdb", "--mode", "saxs", "--length", "0", "--log-level", "ERROR"]),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(1)


def test_main_ai_interpolate_missing_args() -> None:
    """Test ai interpolate mode failure with missing start/end PDB."""
    with (
        patch(
            "sys.argv",
            ["synth-pdb", "--mode", "ai", "--ai-op", "interpolate", "--log-level", "ERROR"],
        ),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(1)


def test_main_ai_missing_op() -> None:
    """Test ai mode failure when --ai-op is missing."""
    with (
        patch("sys.argv", ["synth-pdb", "--mode", "ai", "--log-level", "ERROR"]),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(1)


def test_main_ai_cluster_missing_input() -> None:
    """Test ai cluster mode failure with missing input pattern."""
    with (
        patch(
            "sys.argv",
            ["synth-pdb", "--mode", "ai", "--ai-op", "cluster", "--log-level", "ERROR"],
        ),
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_exit.assert_called_with(1)


def test_main_dataset_invalid_format() -> None:
    """Test dataset mode with an unsupported format (argparse catch)."""
    with (
        patch(
            "sys.argv",
            ["synth-pdb", "--mode", "dataset", "--output", "tmp", "--dataset-format", "invalid"],
        ),
        patch("sys.exit") as mock_exit,
        patch("synth_pdb.main.logger"),  # suppress usage print
    ):
        main()
        # argparse exit code 2 is typical for invalid choice
        mock_exit.assert_called_with(2)


def test_main_docking_happy_path() -> None:
    """Test docking mode successful execution."""
    with (
        patch(
            "sys.argv",
            ["synth-pdb", "--mode", "docking", "--input-pdb", "in.pdb", "--output", "out.pqr"],
        ),
        patch("synth_pdb.main.DockingPrep") as mock_prep_class,
    ):
        mock_prep_instance = mock_prep_class.return_value
        mock_prep_instance.write_pqr.return_value = True
        main()
        mock_prep_instance.write_pqr.assert_called_once()


def test_main_pymol_happy_path() -> None:
    """Test pymol mode successful execution."""
    with (
        patch(
            "sys.argv",
            [
                "synth-pdb",
                "--mode",
                "pymol",
                "--input-pdb",
                "in.pdb",
                "--input-nef",
                "in.nef",
                "--output-pml",
                "out.pml",
            ],
        ),
        patch("synth_pdb.nef_io.read_nef_restraints", return_value=[]),
        patch("synth_pdb.visualization.generate_pymol_script") as mock_gen,
    ):
        main()
        mock_gen.assert_called_once()
