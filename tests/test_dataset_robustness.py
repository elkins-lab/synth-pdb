import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synth_pdb.dataset import (
    DatasetGenerator,
    _generate_single_sample_npz_task,
    _generate_single_sample_task,
)


@pytest.fixture
def temp_output_dir() -> Generator[str, None, None]:
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def test_generate_single_sample_npz_task_invalid_aa(temp_output_dir: str) -> None:
    """Verify that unknown residues are handled gracefully during NPZ generation."""
    # Setup directories
    (Path(temp_output_dir) / "train").mkdir()

    # Mock generate_pdb_content to return a PDB with an unknown residue
    mock_pdb = (
        "ATOM      1  N   XXX A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  XXX A   1       0.000   1.500   0.000  1.00  0.00           C\n"
        "ATOM      3  C   XXX A   1       1.500   1.500   0.000  1.00  0.00           C\n"
        "ATOM      4  O   XXX A   1       1.500   2.500   0.000  1.00  0.00           O\n"
    )

    with patch("synth_pdb.dataset.generate_pdb_content", return_value=mock_pdb):
        args: tuple[Any, ...] = ("test_id", 1, "alpha", "train", temp_output_dir, "npz")
        result = _generate_single_sample_npz_task(args)

        assert result["success"] is True
        # Check the saved NPZ file
        npz_path = Path(temp_output_dir) / "train" / "test_id.npz"
        data = np.load(npz_path)
        # Sequence one-hot for XXX should be all zeros since it's unknown
        assert np.sum(data["sequence"]) == 0.0


def test_generate_single_sample_task_failure(temp_output_dir: str) -> None:
    """Verify that task failure returns success=False and an error message."""
    # Setup directories
    (Path(temp_output_dir) / "train").mkdir()

    with patch(
        "synth_pdb.dataset.generate_pdb_content", side_effect=Exception("Simulated Failure")
    ):
        args: tuple[Any, ...] = ("test_id", 10, "alpha", "train", temp_output_dir, "pdb")
        result = _generate_single_sample_task(args)

        assert result["success"] is False
        assert "Simulated Failure" in result["error"]


def test_dataset_generator_full_manifest_update(temp_output_dir: str) -> None:
    """Test that the manifest is correctly updated even if some samples fail."""
    generator = DatasetGenerator(
        output_dir=temp_output_dir, num_samples=2, max_workers=1, dataset_format="pdb"
    )

    # Mock futures that behave correctly
    mock_f0 = MagicMock()
    mock_f0.result.return_value = {
        "success": True,
        "sample_id": "synth_000000",
        "length": 10,
        "conformation": "alpha",
        "split": "train",
        "pdb_path": "train/s0.pdb",
        "cmap_path": "train/s0.casp",
    }

    mock_f1 = MagicMock()
    mock_f1.result.return_value = {"success": False, "sample_id": "synth_000001", "error": "Fail"}

    # Patch the executor and as_completed
    with (
        patch("concurrent.futures.ProcessPoolExecutor") as mock_exec_class,
        patch(
            "synth_pdb.dataset._generate_single_sample_task",
            side_effect=[mock_f0.result(), mock_f1.result()],
        ),
    ):
        mock_executor = mock_exec_class.return_value.__enter__.return_value
        # Map submit calls to our mock futures
        mock_executor.submit.side_effect = [mock_f0, mock_f1]

        with patch("concurrent.futures.as_completed", return_value=[mock_f0, mock_f1]):
            generator.generate()

            manifest_path = Path(temp_output_dir) / "dataset_manifest.csv"
            with open(manifest_path) as f:
                lines = f.readlines()

            # Header + 1 successful row
            assert len(lines) == 2
            assert "synth_000000" in lines[1]
            assert "synth_000001" not in "".join(lines)


def test_prepare_directories_already_exists(temp_output_dir: str) -> None:
    """Test that prepare_directories handles existing directories and manifest."""
    generator = DatasetGenerator(output_dir=temp_output_dir)
    generator.prepare_directories()
    # Call again, should not crash
    generator.prepare_directories()

    assert (Path(temp_output_dir) / "train").exists()
    assert (Path(temp_output_dir) / "dataset_manifest.csv").exists()


def test_npz_task_missing_cb(temp_output_dir: str) -> None:
    """Verify that residues without CB (like GLY) are handled correctly in NPZ generation."""
    # Setup directories
    (Path(temp_output_dir) / "train").mkdir()

    # Mock generate_pdb_content to return a GLY (no CB)
    mock_pdb = (
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       0.000   1.500   0.000  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       1.500   1.500   0.000  1.00  0.00           C\n"
        "ATOM      4  O   GLY A   1       1.500   2.500   0.000  1.00  0.00           O\n"
    )

    with patch("synth_pdb.dataset.generate_pdb_content", return_value=mock_pdb):
        args: tuple[Any, ...] = ("test_gly", 1, "alpha", "train", temp_output_dir, "npz")
        result = _generate_single_sample_npz_task(args)

        assert result["success"] is True
        npz_path = Path(temp_output_dir) / "train" / "test_gly.npz"
        data = np.load(npz_path)
        # CB is at index 4, should be [0,0,0]
        assert np.all(data["coords"][0, 4] == 0.0)


def test_dataset_generator_multiprocessing_fallback(temp_output_dir: str) -> None:
    """Verify that DatasetGenerator falls back to sequential execution if multiprocessing fails."""
    # num_samples=1 to keep it simple
    generator = DatasetGenerator(output_dir=temp_output_dir, num_samples=1, max_workers=2)

    # Mock result
    mock_result = {
        "success": True,
        "sample_id": "synth_000000",
        "length": 5,
        "conformation": "alpha",
        "split": "train",
        "pdb_path": "train/s0.pdb",
        "cmap_path": "train/s0.casp",
    }

    # 1. Mock ProcessPoolExecutor to raise PermissionError on entry
    # 2. Mock the sequential task function to return success
    with (
        patch(
            "concurrent.futures.ProcessPoolExecutor",
            side_effect=PermissionError("Semaphore failure"),
        ),
        patch("synth_pdb.dataset._generate_single_sample_task", return_value=mock_result),
    ):
        generator.generate()

    # Verify manifest was updated correctly by the sequential fallback
    manifest_path = Path(temp_output_dir) / "dataset_manifest.csv"
    assert manifest_path.exists()
    import pandas as pd

    df = pd.read_csv(manifest_path)
    assert len(df) == 1
    assert df.iloc[0]["id"] == "synth_000000"
