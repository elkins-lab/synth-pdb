#!/usr/bin/env python3

"""
Hardware Acceleration Benchmark for synth-pdb.

This script loops through all available OpenMM platforms (CPU, Metal, CUDA, OpenCL)
and precision modes to measure performance and numerical consistency.
Useful for Apple Silicon and NVIDIA server optimization.
"""

import time
import logging
import argparse
from synth_pdb.generator import PeptideGenerator
from synth_pdb.physics import HAS_OPENMM

if HAS_OPENMM:
    import openmm as mm

# Configure logging to be minimal
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(length: int = 50, iterations: int = 100):
    if not HAS_OPENMM:
        print("❌ Error: OpenMM is not installed. Cannot run benchmark.")
        return

    print("=" * 60)
    print("🚀 synth-pdb Hardware Acceleration Benchmark")
    print(f"Peptide Length: {length} residues")
    print(f"Minimization Limit: {iterations} iterations")
    print("=" * 60)

    # 1. Generate a stable starting structure once
    gen = PeptideGenerator()
    # Use a fixed seed for reproducibility across platforms
    result = gen.generate(length=length, seed=42)
    start_pdb = result.pdb

    # 2. Detect all available platforms
    platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    print(f"Available Platforms: {', '.join(platforms)}")
    print("-" * 60)
    print(f"{'Platform':<15} | {'Precision':<10} | {'Time (s)':<10} | {'Energy (kJ/mol)':<15}")
    print("-" * 60)

    results = []

    for platform_name in platforms:
        # Precision modes to test per platform
        precisions = [None]  # None = default
        if platform_name in ["CUDA", "OpenCL", "Metal"]:
            precisions = ["single", "mixed", "double"]

        for precision in precisions:
            label = precision if precision else "default"

            try:
                # We use a temporary file for the minimization
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f_in:
                    f_in.write(start_pdb)
                    in_path = f_in.name

                out_path = "bench_out.pdb"

                # Time the minimization
                start_t = time.perf_counter()

                # Use current module level imports
                from synth_pdb.physics import EnergyMinimizer

                minimizer = EnergyMinimizer(
                    platform_name=platform_name,
                    precision=precision,
                    disable_cache=True,  # Force new context for clean benchmark
                )

                # Run minimization
                # Note: We must ensure we're calling the right method based on the latest API
                success = minimizer.minimize(in_path, out_path, max_iterations=iterations)

                end_t = time.perf_counter()
                elapsed = end_t - start_t

                if success:
                    print(f"{platform_name:<15} | {label:<10} | {elapsed:>8.3f} | {'Success':<15}")
                    results.append((platform_name, label, elapsed))
                else:
                    print(f"{platform_name:<15} | {label:<10} | {'FAILED':>8} | {'N/A':<15}")

                # Cleanup
                if os.path.exists(in_path):
                    os.unlink(in_path)
                if os.path.exists(out_path):
                    os.unlink(out_path)

            except Exception as e:
                error_msg = str(e)
                print(f"{platform_name:<15} | {label:<10} | {'ERROR':>8} | {error_msg[:30]}...")

    print("-" * 60)
    if results:
        # Find the winner
        results.sort(key=lambda x: x[2])
        best = results[0]
        print(f"🏆 WINNER: {best[0]} ({best[1]}) at {best[2]:.3f} seconds")
    print("\n💡 Tip: If a platform fails, check your OpenMM installation and drivers.")
    print("💡 Run with: PYTHONPATH=. python scripts/benchmark_hardware.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenMM platforms.")
    parser.add_argument("--length", type=int, default=50, help="Peptide length (default: 50)")
    parser.add_argument("--iterations", type=int, default=100, help="Max iterations (default: 100)")
    args = parser.parse_args()

    run_benchmark(length=args.length, iterations=args.iterations)
