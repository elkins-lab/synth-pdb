from unittest.mock import patch

import pytest

from synth_pdb.main import main


class TestClustering:
    """Test suite for the new structural clustering feature."""

    def test_clustering_full_workflow(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that we can generate structures and then cluster them via CLI."""
        # 1. Setup directories
        base_dir = tmp_path
        decoys_dir = base_dir / "decoys"
        clusters_dir = base_dir / "clusters"
        decoys_dir.mkdir()

        # 2. Generate an ensemble of structures
        # We use a short sequence to speed up the test
        gen_args = [
            "synth_pdb",
            "--mode",
            "decoys",
            "--sequence",
            "ACDEFGHIKL",
            "--n-decoys",
            "6",
            "--output",
            str(decoys_dir),
        ]

        with patch("sys.argv", gen_args):
            # We wrap in try/except because main() might call sys.exit
            try:
                main()
            except SystemExit as e:
                if e.code != 0:
                    raise

        # Check that decoys were generated
        generated_files = list(decoys_dir.glob("*.pdb"))
        assert len(generated_files) >= 6

        # 3. Run structural clustering
        input_pattern = str(decoys_dir / "*.pdb")
        cluster_args = [
            "synth_pdb",
            "--mode",
            "ai",
            "--ai-op",
            "cluster",
            "--input-pattern",
            input_pattern,
            "--n-clusters",
            "3",
            "--output",
            str(clusters_dir),
            "--seed",
            "42",
        ]

        with patch("sys.argv", cluster_args):
            try:
                main()
            except SystemExit as e:
                if e.code != 0:
                    raise

        # 4. Validate results
        # Should have exactly 3 medoid files
        medoids = list(clusters_dir.glob("cluster_*_medoid.pdb"))
        assert len(medoids) == 3

        # Verify that the medoids are valid PDB files (non-empty)
        for m in medoids:
            assert m.stat().st_size > 0

    def test_clustering_insufficient_files(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that n_clusters is reduced if file count is too low."""
        base_dir = tmp_path
        one_pdb = base_dir / "one.pdb"

        # Create one dummy PDB file
        with open(one_pdb, "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            )

        clusters_dir = base_dir / "clusters_small"
        cluster_args = [
            "synth_pdb",
            "--mode",
            "ai",
            "--ai-op",
            "cluster",
            "--input-pattern",
            str(one_pdb),
            "--n-clusters",
            "5",  # More than 1
            "--output",
            str(clusters_dir),
        ]

        with patch("sys.argv", cluster_args):
            try:
                main()
            except SystemExit as e:
                if e.code != 0:
                    raise

        # Should only have 1 medoid
        medoids = list(clusters_dir.glob("cluster_*_medoid.pdb"))
        assert len(medoids) == 1
