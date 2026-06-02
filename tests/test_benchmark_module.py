import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from synth_pdb.benchmark import (
    StructureResult,
    BenchmarkResults,
    run_benchmark,
    _extract_sequence,
    _shifts_to_arrays,
)


class TestBenchmarkModule(unittest.TestCase):
    def test_structure_result_init(self) -> None:
        """Test initialization of StructureResult."""
        res = StructureResult(
            sequence="ACDEF",
            length=5,
            conformation="alpha",
            tm_score=0.95,
            gdt_ts=0.9,
            lddt_mean=0.85,
            rmsd=1.2,
            shift_rmsd=0.5,
            gnn_score_ref=0.9,
            gnn_score_pred=0.88,
            predictor_time_s=2.5,
        )
        self.assertEqual(res.sequence, "ACDEF")
        self.assertEqual(res.length, 5)
        self.assertEqual(res.tm_score, 0.95)
        self.assertEqual(res.error, "")

    def test_benchmark_results_summary(self) -> None:
        """Test BenchmarkResults summary generation."""
        r1 = StructureResult(
            tm_score=0.9, gdt_ts=0.8, lddt_mean=0.7, rmsd=1.5, shift_rmsd=0.4, gnn_score_pred=0.8
        )
        r2 = StructureResult(
            tm_score=0.4, gdt_ts=0.3, lddt_mean=0.2, rmsd=5.0, shift_rmsd=1.2, gnn_score_pred=0.4
        )

        results = BenchmarkResults(
            results=[r1, r2], predictor="MockPredictor", n_structures=2, n_success=2
        )

        summary = results.summary()
        self.assertIn("MockPredictor", summary)
        self.assertIn("TM-score   mean=0.650", summary)
        self.assertIn("Structures with TM-score > 0.5 (same fold): 1/2 (50%)", summary)

    def test_benchmark_results_empty_summary(self) -> None:
        """Test summary when all structures failed."""
        results = BenchmarkResults(predictor="FailedModel", n_structures=5, n_success=0)
        summary = results.summary()
        self.assertEqual(summary, "Benchmark 'FailedModel': 0/5 succeeded.")

    def test_benchmark_results_to_csv(self) -> None:
        """Test CSV export."""
        import tempfile

        r1 = StructureResult(sequence="AAA", tm_score=1.0)
        results = BenchmarkResults(results=[r1], n_structures=1, n_success=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path_str = os.path.join(tmp_dir, "subdir", "test_bench.csv")
            results.to_csv(tmp_path_str)
            self.assertTrue(os.path.exists(tmp_path_str))

            df = pd.read_csv(tmp_path_str)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["sequence"], "AAA")
            self.assertEqual(df.iloc[0]["tm_score"], 1.0)

    def test_benchmark_results_to_dataframe(self) -> None:
        """Test conversion to pandas DataFrame."""
        r1 = StructureResult(sequence="AAA", tm_score=1.0)
        results = BenchmarkResults(results=[r1], n_structures=1, n_success=1)
        df = results.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["sequence"], "AAA")

    def test_extract_sequence(self) -> None:
        """Test sequence extraction from PDB string."""
        pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.350   0.000  1.00  0.00           C\n"
            "ATOM      4  N   GLY A   2       3.333   1.450   0.000  1.00  0.00           N\n"
            "ATOM      5  CA  GLY A   2       3.880   2.800   0.000  1.00  0.00           C\n"
            "TER\n"
        )
        seq = _extract_sequence(pdb_content)
        self.assertEqual(seq, "AG")

    def test_shifts_to_arrays(self) -> None:
        """Test normalization of chemical shift data."""
        # Dict format
        d1 = {"H": [1.0, 2.0], "C": [3.0, 4.0]}
        arrs1 = _shifts_to_arrays(d1)
        self.assertTrue(np.array_equal(arrs1["H"], [1.0, 2.0]))

        # List of dicts format
        d2 = [{"H": 1.0, "C": 3.0}, {"H": 2.0, "C": 4.0}]
        arrs2 = _shifts_to_arrays(d2)
        self.assertTrue(np.array_equal(arrs2["H"], [1.0, 2.0]))
        self.assertTrue(np.array_equal(arrs2["C"], [3.0, 4.0]))

    def test_run_benchmark_mock_predictor(self) -> None:
        """Test run_benchmark with a simple mock predictor."""

        def mock_predictor(seq: str) -> str:
            # Return a valid-ish PDB with CA atoms for the given sequence length
            lines = []
            for i, aa in enumerate(seq):
                lines.append(
                    f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {i * 3.8:8.3f}   0.000   0.000  1.00  0.00           C"
                )
            return "\n".join(lines)

        # Run small benchmark
        results = run_benchmark(
            n_structures=2,
            lengths=[5],
            conformations=["alpha"],
            predictor=mock_predictor,
            compute_shifts=False,
            compute_gnn=False,
        )

        self.assertEqual(results.n_structures, 2)
        self.assertEqual(results.n_success, 2)
        self.assertEqual(len(results.results), 2)
        self.assertTrue(np.isfinite(results.results[0].tm_score))

    def test_run_benchmark_error_handling(self) -> None:
        """Test that run_benchmark records errors from failing predictors."""

        def failing_predictor(seq: str) -> str:
            raise RuntimeError("Inference failed")

        results = run_benchmark(
            n_structures=1,
            lengths=[5],
            predictor=failing_predictor,
            compute_shifts=False,
            compute_gnn=False,
        )

        self.assertEqual(results.n_success, 0)
        self.assertIn("Inference failed", results.results[0].error)


if __name__ == "__main__":
    unittest.main()
