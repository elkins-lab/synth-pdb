import os
import shutil
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from synth_pdb.main import main


class TestDatasetReproducibility:
    """Tests that bulk dataset generation is reproducible when using a seed."""

    def test_dataset_seed_reproducibility(self, tmp_path: Path) -> None:
        """Verify that the same seed produces identical manifests (and thus identical samples)."""
        ds_path1 = tmp_path / "ds1"
        ds_path2 = tmp_path / "ds2"

        # 1. Generate first dataset
        # We use a small number of samples for speed
        test_args1 = [
            "synth_pdb",
            "--mode",
            "dataset",
            "--num-samples",
            "3",
            "--seed",
            "42",
            "--output",
            str(ds_path1),
        ]

        # Mock cpu_count to 1 to force sequential execution and avoid PermissionError
        with patch("sys.argv", test_args1), patch("multiprocessing.cpu_count", return_value=1):
            main()

        manifest1_path = ds_path1 / "dataset_manifest.csv"
        assert manifest1_path.exists()
        df1 = pd.read_csv(manifest1_path)

        # 2. Generate second dataset with same seed
        test_args2 = [
            "synth_pdb",
            "--mode",
            "dataset",
            "--num-samples",
            "3",
            "--seed",
            "42",
            "--output",
            str(ds_path2),
        ]

        with patch("sys.argv", test_args2), patch("multiprocessing.cpu_count", return_value=1):
            main()

        manifest2_path = ds_path2 / "dataset_manifest.csv"
        assert manifest2_path.exists()
        df2 = pd.read_csv(manifest2_path)

        # Compare manifests (excluding paths as they are absolute in the manifest sometimes)
        # Actually paths in manifest are relative to output_dir
        cols_to_compare = ["id", "length", "conformation", "split"]
        pd.testing.assert_frame_equal(df1[cols_to_compare], df2[cols_to_compare])

        # 3. Generate third dataset with different seed
        ds_path3 = tmp_path / "ds3"
        test_args3 = [
            "synth_pdb",
            "--mode",
            "dataset",
            "--num-samples",
            "3",
            "--seed",
            "43",
            "--output",
            str(ds_path3),
        ]

        with patch("sys.argv", test_args3), patch("multiprocessing.cpu_count", return_value=1):
            main()

        manifest3_path = ds_path3 / "dataset_manifest.csv"
        df3 = pd.read_csv(manifest3_path)

        # It's statistically possible but extremely unlikely to get same result
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(df1[cols_to_compare], df3[cols_to_compare])
