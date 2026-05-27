import gc
import os

import numpy as np
import psutil
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM


@pytest.mark.skipif(not HAS_OPENMM, reason="Memory leak tests for physics require OpenMM")
class TestMemoryStability:
    """
    Test suite to monitor memory usage during high-volume generation.
    Focuses on identifying leaks in OpenMM Contexts or file handles.
    """

    def test_minimization_memory_leak(self) -> None:
        """
        Generate multiple peptides with energy minimization and monitor RSS memory.
        We expect memory to stabilize after an initial ramp-up.
        """
        process = psutil.Process(os.getpid())

        def get_mem_mb() -> float:
            gc.collect()
            return float(process.memory_info().rss / (1024 * 1024))

        sequence = "ALA-GLY-SER-LEU-GLU"
        n_iterations = 50  # 50 is enough to see a trend without being too slow

        # 1. Warm-up (OpenMM loads libraries and kernels on first use)
        # We use a longer warm-up to ensure stable baseline
        for _ in range(10):
            generate_pdb_content(
                sequence_str=sequence, minimize_energy=True, minimization_max_iter=10
            )

        initial_mem = get_mem_mb()
        print(f"\nInitial Memory: {initial_mem:.2f} MB")

        mem_history = []
        for i in range(n_iterations):
            generate_pdb_content(
                sequence_str=sequence,
                minimize_energy=True,
                minimization_max_iter=20,  # Small but non-zero
            )

            if (i + 1) % 10 == 0:
                # Rigorous collection before measurement
                for _ in range(3):
                    gc.collect()
                current_mem = get_mem_mb()
                mem_history.append(current_mem)
                print(
                    f"Iteration {i + 1}: {current_mem:.2f} MB (Delta: {current_mem - initial_mem:.2f} MB)"
                )

        final_mem = get_mem_mb()
        total_delta = final_mem - initial_mem

        # OpenMM can have some non-linear initialization, so we allow a generous 50MB growth
        # for a batch of 50, but if it's a true leak (e.g. 2MB per context), 50 iter = 100MB+.
        # In a leak-free environment, delta should be nearly 0 after warm-up.
        assert total_delta < 50.0, (
            f"Memory leak detected: {total_delta:.2f} MB growth over {n_iterations} iterations"
        )

        # Also check if the trend is strictly increasing (linear growth)
        if len(mem_history) > 3:
            # Simple linear regression slope check
            x = np.arange(len(mem_history))
            slope = np.polyfit(x, mem_history, 1)[0]
            # Slope should be near zero.
            # We allow 3.0 MB per checkpoint (0.3 MB/iteration) to account for
            # OS-level RSS fluctuations and Python metadata accumulation.
            assert slope < 3.0, (
                f"Significant linear memory growth detected: slope={slope:.4f} MB per 10 iterations"
            )

    def test_batch_generator_memory_stability(self) -> None:
        """Monitor memory usage for large batch generation."""
        from synth_pdb.batch_generator import BatchedGenerator

        process = psutil.Process(os.getpid())

        def get_mem_mb() -> float:
            gc.collect()
            return float(process.memory_info().rss / (1024 * 1024))

        initial_mem = get_mem_mb()
        sequence = "ALA" * 20
        n_iterations = 20
        batch_size = 100

        for _ in range(n_iterations):
            bg = BatchedGenerator(sequence, n_batch=batch_size, full_atom=True)
            _ = bg.generate_batch()

        final_mem = get_mem_mb()
        delta = final_mem - initial_mem
        # Batch generation with templates can cache things, but it shouldn't grow forever.
        assert delta < 30.0, f"Batch generator memory growth: {delta:.2f} MB"
