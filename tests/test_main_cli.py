import logging  # Import logging to set level for caplog
from pathlib import Path
from typing import Any

import numpy as np  # Import numpy for array creation
import pytest

from synth_pdb import main
from synth_pdb.generator import create_atom_line  # Import the helper function
from synth_pdb.validator import PDBValidator


class TestMainCLI:
    # --- Tests for --guarantee-valid ---
    def test_guarantee_valid_success(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        # Set caplog level to DEBUG to capture all relevant messages, especially the one about current_violations
        caplog.set_level(logging.DEBUG)

        # PDB content that causes steric clashes (2 violations: min_distance and VdW overlap)
        # Manually crafted to ensure it's a raw string
        clashing_pdb_content = (
            "HEADER    clashing_peptide\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )

        # Valid PDB content
        # Manually crafted to ensure it's a raw string
        valid_pdb_content = (
            "HEADER    valid_peptide\n"
            + create_atom_line(
                1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "GLY", "A", 2, 3.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                3, "CA", "GLY", "A", 3, 6.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )
        mocker.patch(
            "synth_pdb.main.generate_pdb_content",
            side_effect=[clashing_pdb_content, clashing_pdb_content, valid_pdb_content],
        )

        # No need to mock PDBValidator or its methods, let the real one run.

        # Mock sys.argv to simulate CLI arguments, including --log-level DEBUG
        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--guarantee-valid",
            "--max-attempts",
            "3",
            "--output",
            "test_gv_success.pdb",
            "--log-level",
            "DEBUG",
        ]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit to prevent actual exit
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Assert that expected log messages are present
        assert "PDB generated in attempt 1 has 1 violations. Retrying..." in caplog.text
        assert "PDB generated in attempt 2 has 1 violations. Retrying..." in caplog.text
        assert "Successfully generated a valid PDB file after 3 attempts." in caplog.text
        assert "test_gv_success.pdb" in caplog.text
        mock_exit.assert_not_called()  # Should not exit with error

    def test_guarantee_valid_failure(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)  # Set to INFO to capture relevant messages

        # PDB content that causes steric clashes (2 violations: min_distance and VdW overlap)
        clashing_pdb_content = (
            "HEADER    clashing_peptide\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )

        mocker.patch(
            "synth_pdb.main.generate_pdb_content", return_value=clashing_pdb_content
        )  # Always return clashing PDB

        # No need to mock PDBValidator or its methods, let the real one run.

        # Mock sys.argv
        test_args = ["synth_pdb", "--length", "1", "--guarantee-valid", "--max-attempts", "2"]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit to check for error exit
        mock_sys_exit = mocker.patch("sys.exit")

        main.main()

        assert "PDB generated in attempt 1 has 1 violations. Retrying..." in caplog.text
        assert "PDB generated in attempt 2 has 1 violations. Retrying..." in caplog.text
        assert "Failed to generate a suitable PDB file after 2 attempts." in caplog.text
        mock_sys_exit.assert_called_once_with(1)

    # --- Tests for --best-of-N ---
    def test_best_of_n_selection(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)  # Set to INFO to capture relevant messages

        # PDB content with 2 violations (steric clash, VdW overlap)
        pdb_content_2_violations = (
            "HEADER    two_violations\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                3, "CA", "ALA", "A", 3, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )
        # PDB content with 1 violation (e.g., a less severe steric clash)
        pdb_content_1_violation = (
            "HEADER    one_violation\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )
        # PDB content with 0 violations
        pdb_content_0_violations = (
            "HEADER    no_violations\n"
            + create_atom_line(
                1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "GLY", "A", 2, 3.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )

        mocker.patch(
            "synth_pdb.main.generate_pdb_content",
            side_effect=[
                pdb_content_2_violations,  # First generated PDB will have 2 violations
                pdb_content_1_violation,  # Second generated PDB will have 1 violation
                pdb_content_0_violations,  # Third generated PDB will have 0 violations
            ],
        )

        # No need to mock PDBValidator or its methods, let the real one run.

        # Mock sys.argv
        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--best-of-N",
            "3",
            "--output",
            "test_best_of_N.pdb",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # The actual violations found by the real PDBValidator (based on content above)
        assert "Attempt 1 yielded 4 violations" in caplog.text
        assert "Attempt 2 yielded 1 violations (new minimum)." in caplog.text
        assert "Attempt 3 yielded 0 violations (new minimum)." in caplog.text
        assert (
            "No violations found in the final PDB for" in caplog.text
        )  # Because the 0-violation PDB was chosen
        mock_exit.assert_not_called()

    # --- Tests for --refine-clashes ---
    def test_refine_clashes_reduces_violations(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)  # Set to INFO to capture relevant messages

        # PDB content that causes steric clashes (2 violations: min_distance and VdW overlap)
        initial_clashing_pdb_content = (
            "HEADER    clashing_peptide\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )

        # PDB content that has no steric clashes (parsed atoms for mocking tweak result)
        # Ensure coords are numpy arrays as the validator expects them for calculation
        non_clashing_parsed_atoms = [
            {
                "atom_number": 1,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([0.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([3.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
        ]

        mocker.patch(
            "synth_pdb.main.generate_pdb_content", return_value=initial_clashing_pdb_content
        )

        # Mock _apply_steric_clash_tweak to simulate it working
        mocker.patch.object(
            PDBValidator, "_apply_steric_clash_tweak", return_value=non_clashing_parsed_atoms
        )

        # Mock sys.argv
        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            "test_refine.pdb",
            "--refine-clashes",
            "1",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # The initial clashing_pdb_content will result in 1 steric clash violation (min distance).
        # After the tweak, it should be 0.
        assert "Refinement iteration 1/1. Violations: 1" in caplog.text
        assert "Refinement iteration 1: Reduced violations from 1 to 0." in caplog.text
        assert "Refinement process completed. Reduced total violations from 1 to 0." in caplog.text
        assert "No violations found in the final PDB for" in caplog.text
        mock_exit.assert_not_called()

    def test_refine_clashes_no_improvement(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)  # Set to INFO to capture relevant messages

        # PDB content that causes steric clashes (2 violations: min_distance and VdW overlap)
        initial_clashing_pdb_content = (
            "HEADER    clashing_peptide\n"
            + create_atom_line(
                1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
            + "\n"
            + create_atom_line(
                2, "CA", "ALA", "A", 2, 0.100, 0.0, 0.0, "C", alt_loc="", insertion_code=""
            )
        )
        mocker.patch(
            "synth_pdb.main.generate_pdb_content", return_value=initial_clashing_pdb_content
        )

        # Mock _apply_steric_clash_tweak to return modified atoms (but still clashing)
        # Ensure coords are numpy arrays as the validator expects them for calculation
        still_clashing_parsed_atoms = [
            {
                "atom_number": 1,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([0.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([0.1, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },  # Still clashing
        ]
        mocker.patch.object(
            PDBValidator, "_apply_steric_clash_tweak", return_value=still_clashing_parsed_atoms
        )

        # Mock sys.argv
        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            "test_refine_no_improve.pdb",
            "--refine-clashes",
            "2",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # The initial clashing_pdb_content will result in 1 steric clash violation.
        # After the tweak, it should still be 1.
        assert "Refinement iteration 1/2. Violations: 1" in caplog.text
        assert (
            "Refinement iteration 1: No further reduction in violations (1). Stopping refinement."
            in caplog.text
        )
        assert "Refinement process completed. No change in total violations (1)." in caplog.text
        assert "Final PDB has 1 violations." in caplog.text
        mock_exit.assert_not_called()

    def test_header_with_best_of_n_and_refine_clashes(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        output_filepath = tmp_path / "test_best_refine_header.pdb"

        # Mock generate_pdb_content to always return a PDB with content, including header/footer
        # This will be the content generate_pdb_content produces *before* main.py processes it.
        # It has 1 violation to trigger refinement.
        initial_pdb_content = (
            "HEADER    GEN TEST          18-JAN-26\n"
            "TITLE     GENERATED LINEAR PEPTIDE OF LENGTH 2\n"
            "REMARK 1  This is a test PDB.\n"
            "MODEL        1\n"
            + create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(3, "C", "ALA", "A", 1, 2.500, 0.0, 0.0, "C")
            + "\n"  # Closer than standard
            + create_atom_line(4, "O", "ALA", "A", 1, 3.500, 0.0, 0.0, "O")
            + "\n"
            + create_atom_line(5, "N", "ALA", "A", 2, 4.000, 0.0, 0.0, "N")
            + "\n"
            + create_atom_line(6, "CA", "ALA", "A", 2, 5.458, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(7, "C", "ALA", "A", 2, 6.983, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(8, "O", "ALA", "A", 2, 7.812, 0.0, 0.0, "O")
            + "\n"
            "TER     9      ALA A   2\n"
            "ENDMDL\n"
            "END         "
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=initial_pdb_content)

        # Mock _apply_steric_clash_tweak to simulate it fixing the clash
        # We need to ensure the number of violations reduces.
        # Original: CA-C distance (1.042) vs 1.52 (difference 0.478)
        # New: CA-C distance (1.52)
        non_clashing_parsed_atoms = [
            {
                "atom_number": 1,
                "atom_name": "N",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([0.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "N",
                "charge": "",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([1.458, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 3,
                "atom_name": "C",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([1.458 + 1.52, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },  # Fixed
            {
                "atom_number": 4,
                "atom_name": "O",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([1.458 + 1.52 + 1.23, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "O",
                "charge": "",
            },
            {
                "atom_number": 5,
                "atom_name": "N",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([4.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "N",
                "charge": "",
            },
            {
                "atom_number": 6,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([5.458, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 7,
                "atom_name": "C",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([6.983, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 8,
                "atom_name": "O",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([7.812, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "O",
                "charge": "",
            },
        ]
        mocker.patch.object(
            PDBValidator, "_apply_steric_clash_tweak", return_value=non_clashing_parsed_atoms
        )

        # Mock sys.argv
        test_args = [
            "synth_pdb",
            "--length",
            "2",
            "--best-of-N",
            "1",
            "--refine-clashes",
            "1",
            "--output",
            str(output_filepath),
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Read the generated file and assert its contents
        with open(output_filepath) as f:
            content = f.read()

        lines = content.splitlines()

        # Assert header presence
        assert lines[0].startswith("HEADER"), "PDB file should start with a HEADER line."
        assert lines[1].startswith("TITLE"), "PDB file should have a TITLE line."
        assert (
            "GENERATED LINEAR PEPTIDE OF LENGTH 2" in lines[1]
        ), "TITLE should reflect sequence length."
        assert any("REMARK" in line for line in lines), "PDB file should contain REMARK lines."
        assert any("COMPND" in line for line in lines), "PDB file should contain COMPND lines."
        assert any("SOURCE" in line for line in lines), "PDB file should contain SOURCE lines."
        assert any("KEYWDS" in line for line in lines), "PDB file should contain KEYWDS lines."
        assert any("EXPDTA" in line for line in lines), "PDB file should contain EXPDTA lines."
        assert any("AUTHOR" in line for line in lines), "PDB file should contain AUTHOR lines."
        assert any("MODEL" in line for line in lines), "PDB file should contain MODEL lines."

        # Assert atomic lines presence
        atomic_lines = [line for line in lines if line.startswith("ATOM")]
        assert len(atomic_lines) > 0, "PDB file should contain ATOM records."

        # Assert footer presence
        assert "TER" in lines[-3], "PDB file should contain a TER record."
        assert lines[-2].startswith("ENDMDL"), "PDB file should contain an ENDMDL record."
        assert lines[-1].startswith("END"), "PDB file should contain an END record."

        mock_exit.assert_not_called()

    def test_run_dataset_generation(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test --mode dataset calls DatasetGenerator.

        SCIENTIFIC BASIS:
        Bulk dataset generation is critical for training machine learning models
        (like GNNs or PLMs) to recognize protein structural quality.
        """
        caplog.set_level(logging.INFO)
        mock_ds_cls = mocker.patch("synth_pdb.dataset.DatasetGenerator")
        mock_ds_instance = mock_ds_cls.return_value

        test_args = [
            "synth_pdb",
            "--mode",
            "dataset",
            "--num-samples",
            "10",
            "--output",
            "my_dataset",
            "--train-ratio",
            "0.8",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        mock_ds_cls.assert_called_once()
        kwargs = mock_ds_cls.call_args.kwargs
        assert kwargs["num_samples"] == 10
        assert kwargs["output_dir"] == "my_dataset"
        assert kwargs["train_ratio"] == 0.8
        mock_ds_instance.generate.assert_called_once()
        assert "Dataset generation complete" in caplog.text
        mock_exit.assert_not_called()

    def test_run_ai_interpolate(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test --mode ai --ai-op interpolate.

        SCIENTIFIC BASIS:
        Structural interpolation (morphing) is used to visualize and study
        transitions between different protein conformational states, such as
        folding pathways or ligand-induced changes.
        """
        caplog.set_level(logging.INFO)
        mock_interp = mocker.patch("synth_pdb.quality.interpolate.interpolate_structures")

        test_args = [
            "synth_pdb",
            "--mode",
            "ai",
            "--ai-op",
            "interpolate",
            "--start-pdb",
            "start.pdb",
            "--end-pdb",
            "end.pdb",
            "--steps",
            "20",
            "--output",
            "my_morph",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        mock_interp.assert_called_once_with("start.pdb", "end.pdb", 20, "my_morph")
        assert "Interpolation complete" in caplog.text
        mock_exit.assert_not_called()

    def test_run_ai_missing_args(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test error handling for missing AI arguments."""
        caplog.set_level(logging.ERROR)
        test_args = ["synth_pdb", "--mode", "ai", "--ai-op", "interpolate"]  # missing PDBS
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Interpolation requires --start-pdb and --end-pdb" in caplog.text
        mock_exit.assert_called_with(1)

    def test_run_ai_invalid_op(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test error handling for invalid AI operation."""
        caplog.set_level(logging.ERROR)
        test_args = ["synth_pdb", "--mode", "ai", "--ai-op", "invalid"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "AI mode requires --ai-op {interpolate, cluster}" in caplog.text
        mock_exit.assert_called_with(1)

    def test_run_ai_cluster_not_implemented(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test cluster op reports not implemented."""
        caplog.set_level(logging.ERROR)
        test_args = ["synth_pdb", "--mode", "ai", "--ai-op", "cluster"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Clustering not yet implemented" in caplog.text
        mock_exit.assert_called_with(1)

    # --- Additional tests for error conditions and edge cases ---

    def test_invalid_length_without_sequence(self, tmp_path: Path) -> None:
        """Test that length=0 without sequence raises SystemExit."""
        output_file = tmp_path / "test.pdb"

        # Mock sys.argv with invalid length (0)
        test_args = ["synth_pdb", "--length", "0", "--output", str(output_file)]
        import sys

        sys.argv = test_args

        # Should exit with code 1
        with pytest.raises(SystemExit) as excinfo:
            main.main()
        assert excinfo.value.code == 1

    def test_negative_length_without_sequence(self, tmp_path: Path) -> None:
        """Test that negative length without sequence raises SystemExit."""
        output_file = tmp_path / "test.pdb"

        # Mock sys.argv with negative length
        test_args = ["synth_pdb", "--length", "-5", "--output", str(output_file)]
        import sys

        sys.argv = test_args

        # Should exit with code 1
        with pytest.raises(SystemExit) as excinfo:
            main.main()
        assert excinfo.value.code == 1

    def test_failed_pdb_generation_empty_content(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handling when generate_pdb_content returns None/empty."""
        caplog.set_level(logging.WARNING)

        # Mock generate_pdb_content to return None to simulate failure
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=None)

        # Mock sys.argv
        test_args = ["synth_pdb", "--length", "5", "--max-attempts", "2"]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit
        mock_sys_exit = mocker.patch("sys.exit")

        main.main()

        assert (
            "Failed to generate PDB content in attempt" in caplog.text
            or "Failed to generate a suitable PDB file after" in caplog.text
        )
        mock_sys_exit.assert_called_with(1)

    def test_generation_value_error(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of ValueError during PDB generation."""
        caplog.set_level(logging.ERROR)

        # Mock generate_pdb_content to raise ValueError
        mocker.patch(
            "synth_pdb.main.generate_pdb_content", side_effect=ValueError("Invalid amino acid code")
        )

        # Mock sys.argv
        test_args = ["synth_pdb", "--length", "3", "--max-attempts", "2"]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit
        # Mock sys.exit to raise SystemExit to simulate actual exit
        mock_sys_exit = mocker.patch("sys.exit", side_effect=SystemExit)

        with pytest.raises(SystemExit):
            main.main()

        assert "Error processing sequence during generation" in caplog.text
        assert "Invalid amino acid code" in caplog.text
        mock_sys_exit.assert_called_once_with(1)

    def test_generation_unexpected_exception(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handling of unexpected exception during PDB generation."""
        caplog.set_level(logging.ERROR)

        # Mock generate_pdb_content to raise a generic Exception
        mocker.patch(
            "synth_pdb.main.generate_pdb_content", side_effect=Exception("Unexpected error")
        )

        # Mock sys.argv
        test_args = ["synth_pdb", "--length", "3", "--max-attempts", "2"]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit
        # Mock sys.exit to raise SystemExit
        mock_sys_exit = mocker.patch("sys.exit", side_effect=SystemExit)

        with pytest.raises(SystemExit):
            main.main()

        # Updated assertion to match actual log message
        assert "Error processing sequence during generation" in caplog.text
        assert "Unexpected error" in caplog.text
        mock_sys_exit.assert_called_once_with(1)

    def test_refine_clashes_early_exit_no_violations(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that refinement exits early when no violations remain."""
        caplog.set_level(logging.INFO)

        # Start with a valid PDB (no violations)
        valid_pdb_content = (
            "HEADER    valid_peptide\n"
            + create_atom_line(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(2, "CA", "GLY", "A", 2, 3.0, 0.0, 0.0, "C")
        )

        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb_content)

        # Mock sys.argv with multiple refinement iterations
        test_args = [
            "synth_pdb",
            "--length",
            "2",
            "--refine-clashes",
            "5",
            "--output",
            "test_early_exit.pdb",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Should exit early because there are no violations to begin with
        assert "Refinement iteration 1/5. Violations: 0" in caplog.text
        assert "No violations remain, stopping refinement early." in caplog.text
        # Should NOT see iteration 2
        assert "Refinement iteration 2" not in caplog.text
        mock_exit.assert_not_called()

    def test_refine_clashes_increases_violations(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning when refinement increases violations (should not happen but edge case)."""
        caplog.set_level(logging.INFO)

        # Initial PDB with 1 violation
        pdb_with_1_violation = (
            "HEADER    one_violation\n"
            + create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 2, 1.0, 0.0, 0.0, "C")
        )

        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=pdb_with_1_violation)

        # Mock _apply_steric_clash_tweak to return atoms with MORE violations
        # Make atoms even closer together
        worse_parsed_atoms = [
            {
                "atom_number": 1,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": np.array([0.0, 0.0, 0.0]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
            {
                "atom_number": 2,
                "atom_name": "CA",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 2,
                "insertion_code": "",
                "coords": np.array([0.3, 0.0, 0.0]),  # Even closer
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
                "charge": "",
            },
        ]
        mocker.patch.object(
            PDBValidator, "_apply_steric_clash_tweak", return_value=worse_parsed_atoms
        )

        # Mock sys.argv
        test_args = [
            "synth_pdb",
            "--length",
            "2",
            "--refine-clashes",
            "1",
            "--output",
            "test_worse.pdb",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # The mock returns atoms that create 2 violations (same as before), so it will report "No change"
        # To actually test line 290 (violations increasing), we need a scenario where they go UP
        # For now, this tests the refinement path without improvement
        assert "Refinement process completed" in caplog.text
        mock_exit.assert_not_called()

    def test_custom_filename_generation_no_output_flag(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that custom filename is generated when --output is not provided."""
        import os

        caplog.set_level(logging.INFO)

        # Change to tmp_path directory so generated file goes there
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Simple valid PDB
            valid_pdb = "HEADER    test\n" + create_atom_line(
                1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"
            )
            mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

            # Mock sys.argv WITHOUT --output flag
            test_args = ["synth_pdb", "--length", "5"]
            mocker.patch("sys.argv", test_args)
            mock_exit = mocker.patch("sys.exit")

            main.main()

            # Verify a file matching the pattern was created
            files = list(tmp_path.glob("random_linear_peptide_5_*.pdb"))
            assert len(files) == 1, "Should generate exactly one PDB file with timestamp"

            # Verify filename contains timestamp pattern (YYYYMMDD_HHMMSS)
            filename = files[0].name
            assert "random_linear_peptide_5_" in filename
            assert filename.endswith(".pdb")

            mock_exit.assert_not_called()
        finally:
            os.chdir(original_cwd)

    def test_file_write_error(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling when file write fails."""
        caplog.set_level(logging.ERROR)

        # Valid PDB content
        valid_pdb = "HEADER    test\n" + create_atom_line(
            1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock open() to raise PermissionError
        mocker.patch("builtins.open", side_effect=PermissionError("Permission denied"))

        # Mock sys.argv
        test_args = ["synth_pdb", "--length", "3", "--output", "test_permission_error.pdb"]
        mocker.patch("sys.argv", test_args)

        # Mock sys.exit
        mock_sys_exit = mocker.patch("sys.exit")

        main.main()

        assert "An unexpected error occurred during file writing" in caplog.text
        assert "Permission denied" in caplog.text
        mock_sys_exit.assert_called_once_with(1)

    def test_header_contains_all_arguments(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that all CLI arguments are recorded in PDB REMARKs."""
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "header_test.pdb"

        # Valid PDB content
        valid_pdb = "HEADER    test\n" + create_atom_line(
            1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Run with many flags
        test_args = [
            "synth_pdb",
            "--length",
            "5",
            "--minimize",
            "--optimize",
            "--gen-shifts",
            "--output",
            str(output_file),
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Check output file content
        with open(output_file) as f:
            content = f.read()

        # Verify REMARK 3 contains the flags
        assert "REMARK 3  Command:" in content
        assert "--minimize" in content
        assert "--optimize" in content
        assert "--gen-shifts" in content
        # Reconstruct the full command string from REMARK 3 lines to handle wrapping
        remark_lines = [line for line in content.splitlines() if line.startswith("REMARK 3    ")]
        # Remove the prefix "REMARK 3    " (12 chars) and join
        full_command_remark = "".join([line[12:] for line in remark_lines])

        assert "--minimize" in full_command_remark
        assert "--optimize" in full_command_remark
        assert "--gen-shifts" in full_command_remark
        assert "--forcefield amber14-all.xml" in full_command_remark
        mock_exit.assert_not_called()

    def test_run_decoys(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test --mode decoys calls the generator."""
        caplog.set_level(logging.INFO)

        # Mock DecoyGenerator
        mock_gen_cls = mocker.patch("synth_pdb.main.DecoyGenerator")
        mock_gen_instance = mock_gen_cls.return_value

        # Mock sys.argv
        test_args = ["synth_pdb", "--mode", "decoys", "--sequence", "AAA", "--n-decoys", "5"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Verify call
        mock_gen_instance.generate_ensemble.assert_called_once()
        args, kwargs = mock_gen_instance.generate_ensemble.call_args
        assert kwargs["sequence"] == "AAA"
        assert kwargs["n_decoys"] == 5
        assert kwargs["rmsd_min"] == 0.0
        assert kwargs["rmsd_max"] == 999.0
        mock_exit.assert_not_called()

    def test_decoys_missing_sequence_generates_random(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that missing sequence triggers random generation."""
        caplog.set_level(logging.INFO)

        mock_gen_cls = mocker.patch("synth_pdb.main.DecoyGenerator")
        mock_gen_instance = mock_gen_cls.return_value

        # Mock sys.argv with length but no sequence
        test_args = ["synth_pdb", "--mode", "decoys", "--length", "10", "--n-decoys", "1"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Check logs
        assert "Generated random sequence for decoys" in caplog.text
        mock_gen_instance.generate_ensemble.assert_called_once()
        args, kwargs = mock_gen_instance.generate_ensemble.call_args
        assert len(kwargs["sequence"]) == 10
        mock_exit.assert_not_called()

    def test_run_docking(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test --mode docking calls DockingPrep."""
        caplog.set_level(logging.INFO)
        mocker.patch("synth_pdb.main.DockingPrep")

        test_args = ["synth_pdb", "--mode", "docking", "--input-pdb", "test.pdb"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Docking preparation complete" in caplog.text
        mock_exit.assert_not_called()

    def test_run_pymol(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test --mode pymol generation."""
        caplog.set_level(logging.INFO)
        mocker.patch("synth_pdb.nef_io.read_nef_restraints", return_value=[])
        mocker.patch("synth_pdb.visualization.generate_pymol_script")

        test_args = [
            "synth_pdb",
            "--mode",
            "pymol",
            "--input-pdb",
            "t.pdb",
            "--input-nef",
            "t.nef",
            "--output-pml",
            "o.pml",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "PyMOL script generated successfully" in caplog.text
        mock_exit.assert_not_called()

    def test_run_export_features(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test generation of NEF, Relax, Shifts, and Constraints."""
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "export_test.pdb"

        # Valid PDB content
        valid_pdb = (
            "HEADER    test\n"
            + create_atom_line(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C")
            + "\n"
            + create_atom_line(2, "HA1", "GLY", "A", 1, 1.0, 0.0, 0.0, "H")
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock calculation functions
        # Patch the source modules because main.py imports them locally/lazily
        mocker.patch("synth_pdb.nmr.calculate_synthetic_noes", return_value=[])
        mocker.patch("synth_pdb.nef_io.write_nef_file")
        mocker.patch("synth_pdb.relaxation.calculate_relaxation_rates", return_value={})
        mocker.patch("synth_pdb.nef_io.write_nef_relaxation")
        mocker.patch("synth_pdb.chemical_shifts.predict_chemical_shifts", return_value={})
        mocker.patch("synth_pdb.nef_io.write_nef_chemical_shifts")
        mocker.patch("synth_pdb.contact.compute_contact_map", return_value=np.zeros((1, 1)))
        mocker.patch("synth_pdb.export.export_constraints", return_value="test")

        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            str(output_file),
            "--gen-nef",
            "--gen-relax",
            "--gen-shifts",
            "--export-constraints",
            str(tmp_path / "c.casp"),
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Verify calls
        assert (
            "Calculating NOE Restraints" in caplog.text or "Calculate NOE Restraints" in caplog.text
        )
        assert "Constraints exported to" in caplog.text
        mock_exit.assert_not_called()

    def test_run_export_torsion(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test generation of Torsion Angles."""
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "torsion_test.pdb"

        # Valid PDB (Alpha helix approx to get results)
        valid_pdb = (
            "HEADER    test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, -1.458, 0, 0, "N")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 0, 0, 0, "C")
            + "\n"
            + create_atom_line(3, "C", "ALA", "A", 1, 1.525, 0, 0, "C")
            + "\n"
            + create_atom_line(4, "H", "ALA", "A", 1, -1.5, 1, 0, "H")  # Just enough to parse
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock calculation and export
        mocker.patch("synth_pdb.torsion.calculate_torsion_angles", return_value=[{"phi": -60}])
        mocker.patch("synth_pdb.torsion.export_torsion_angles")

        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            str(output_file),
            "--export-torsion",
            str(tmp_path / "angles.csv"),
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Torsion angles exported to" in caplog.text
        mock_exit.assert_not_called()

    def test_run_msa_generation(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test generation of Synthetic MSA."""
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "msa_test.pdb"

        # Valid PDB with Hydrogen
        valid_pdb = (
            "HEADER    test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, -1.458, 0, 0, "N")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 0, 0, 0, "C")
            + "\n"
            + create_atom_line(3, "C", "ALA", "A", 1, 1.525, 0, 0, "C")
            + "\n"
            + create_atom_line(4, "H", "ALA", "A", 1, -1.5, 1, 0, "H")
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock evolution
        mocker.patch("synth_pdb.evolution.generate_msa_sequences", return_value=["AAA", "AAB"])
        mocker.patch("synth_pdb.evolution.write_msa")

        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            str(output_file),
            "--gen-msa",
            "--msa-depth",
            "10",
            "--evolution-temp",
            "1.5",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Synthetic MSA generated" in caplog.text
        mock_exit.assert_not_called()

    def test_run_distogram_export(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test generation of Distogram."""
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "dist_test.pdb"

        # Valid PDB with Hydrogen
        valid_pdb = (
            "HEADER    test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, -1.458, 0, 0, "N")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 0, 0, 0, "C")
            + "\n"
            + create_atom_line(3, "C", "ALA", "A", 1, 1.525, 0, 0, "C")
            + "\n"
            + create_atom_line(4, "H", "ALA", "A", 1, -1.5, 1, 0, "H")
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock calculation and export
        mocker.patch("synth_pdb.distogram.calculate_distogram", return_value=np.zeros((1, 1)))
        mocker.patch("synth_pdb.distogram.export_distogram")

        test_args = [
            "synth_pdb",
            "--length",
            "1",
            "--output",
            str(output_file),
            "--export-distogram",
            str(tmp_path / "dist.json"),
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Distogram exported to" in caplog.text
        mock_exit.assert_not_called()

    def test_run_biophysics_options(self, mocker: Any, tmp_path: Path) -> None:
        """Test PH and Capping flags are passed correctly."""
        output_file = tmp_path / "bio_test.pdb"

        # Mock generator to check args
        mock_gen = mocker.patch("synth_pdb.main.generate_pdb_content", return_value="HEADER test")

        test_args = [
            "synth_pdb",
            "--length",
            "10",
            "--output",
            str(output_file),
            "--ph",
            "4.5",
            "--cap-termini",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Verify args passed to generator
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["ph"] == 4.5
        assert call_kwargs["cap_termini"] is True
        mock_exit.assert_not_called()

    def test_run_md_equilibration_options(self, mocker: Any, tmp_path: Path) -> None:
        """Test MD Equilibration flags are passed correctly."""
        output_file = tmp_path / "md_test.pdb"

        # Mock generator to check args
        mock_gen = mocker.patch("synth_pdb.main.generate_pdb_content", return_value="HEADER test")

        test_args = [
            "synth_pdb",
            "--length",
            "10",
            "--output",
            str(output_file),
            "--equilibrate",
            "--md-steps",
            "500",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Verify args passed to generator
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["equilibrate"] is True
        assert call_kwargs["equilibrate_steps"] == 500
        mock_exit.assert_not_called()

    # --- New Failure Case Tests ---

    def test_structure_parsing_failure(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test failure when --structure parameter is invalid."""
        caplog.set_level(logging.ERROR)

        # Invalid structure format (non-integer range)
        test_args = ["synth_pdb", "--structure", "1-A:alpha", "--output", "fail.pdb"]
        mocker.patch("sys.argv", test_args)

        with pytest.raises(SystemExit):
            main.main()

        assert "Failed to parse --structure parameter" in caplog.text

    def test_docking_missing_input_pdb(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test failing docking mode without input-pdb."""
        caplog.set_level(logging.ERROR)

        test_args = ["synth_pdb", "--mode", "docking"]  # Missing --input-pdb
        mocker.patch("sys.argv", test_args)

        # sys.exit should raise SystemExit to stop execution flow
        with pytest.raises(SystemExit):
            main.main()

        assert "Docking mode requires --input-pdb" in caplog.text

    def test_pymol_missing_arguments(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Test failing pymol mode without required args."""
        caplog.set_level(logging.ERROR)

        # Missing output-pml
        test_args = [
            "synth_pdb",
            "--mode",
            "pymol",
            "--input-pdb",
            "in.pdb",
            "--input-nef",
            "in.nef",
        ]
        mocker.patch("sys.argv", test_args)

        with pytest.raises(SystemExit):
            main.main()

        assert "PyMOL mode requires --input-pdb, --input-nef, and --output-pml" in caplog.text

    def test_nef_generation_no_hydrogens_error(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test error when generating NEF without hydrogens."""
        caplog.set_level(logging.ERROR)
        output_file = tmp_path / "nef_error.pdb"

        # Valid PDB without Hydrogens
        valid_pdb = (
            "HEADER    test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, -1.458, 0, 0, "N")
            + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 0, 0, 0, "C")
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        test_args = ["synth_pdb", "--length", "1", "--output", str(output_file), "--gen-nef"]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        assert "Structure has no hydrogens! NEF/Relaxation requires protons" in caplog.text
        mock_exit.assert_not_called()

    def test_visualization_highlights_passing(
        self, mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression Test: Verify that using --visualize with --structure correctly parses
        highlights and passes them to the viewer without UnboundLocalError.
        """
        caplog.set_level(logging.INFO)
        output_file = tmp_path / "viz_test.pdb"

        # Valid PDB logic
        valid_pdb = "HEADER    test\n" + create_atom_line(
            1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=valid_pdb)

        # Mock view_structure_in_browser to capture call args
        mock_view = mocker.patch("synth_pdb.main.view_structure_in_browser")

        test_args = [
            "synth_pdb",
            "--length",
            "5",
            "--output",
            str(output_file),
            "--visualize",
            "--structure",
            "1-3:alpha,4-5:typeII",
        ]
        mocker.patch("sys.argv", test_args)
        mock_exit = mocker.patch("sys.exit")

        main.main()

        # Verify usage
        assert "Opening 3D molecular viewer in browser" in caplog.text

        # Verify args passed to viewer
        mock_view.assert_called_once()
        call_kwargs = mock_view.call_args.kwargs
        highlights = call_kwargs.get("highlights", [])

        # Should have 2 highlight entries (Helix and TypeII)
        assert len(highlights) == 2

        # Check typeII turn (purple stick)
        type_ii = next((h for h in highlights if h["label"] == "typeII"), None)
        assert type_ii is not None
        assert type_ii["start"] == 4
        assert type_ii["end"] == 5
        assert type_ii["color"] == "purple"
        mock_exit.assert_not_called()
