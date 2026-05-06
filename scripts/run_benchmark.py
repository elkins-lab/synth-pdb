"""Run the synth-pdb vs. AI structure prediction benchmark.

Usage
-----
    # ESMFold (default, no API key required)
    python scripts/run_benchmark.py --n-structures 20

    # Custom lengths and conformations
    python scripts/run_benchmark.py --n-structures 50 --lengths 20 30 50 --conformations alpha beta

    # Save results to CSV
    python scripts/run_benchmark.py --n-structures 20 --output benchmark_results.csv

    # Skip chemical shift and GNN scoring for a fast geometry-only run
    python scripts/run_benchmark.py --n-structures 20 --no-shifts --no-gnn

Notes
-----
    ESMFold downloads ~700 MB on first use (cached by HuggingFace).
    Install requirements: pip install transformers accelerate synth-pdb[gnn]
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark AI structure predictors against synth-pdb ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictor",
        default="esmfold",
        choices=["esmfold"],
        help="Structure prediction backend (default: esmfold)",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=20,
        help="Number of test structures to generate and evaluate (default: 20)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[20, 30, 50],
        metavar="L",
        help="Chain lengths to sample from (default: 20 30 50)",
    )
    parser.add_argument(
        "--conformations",
        nargs="+",
        default=["alpha", "beta"],
        choices=["alpha", "beta", "random"],
        metavar="CONF",
        help="Conformations to sample from (default: alpha beta)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save results to a CSV file at this path",
    )
    parser.add_argument(
        "--no-shifts",
        action="store_true",
        help="Skip chemical shift RMSD computation",
    )
    parser.add_argument(
        "--no-gnn",
        action="store_true",
        help="Skip GNN quality scoring",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for reproducible structure generation (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Run benchmark ──────────────────────────────────────────────────
    try:
        from synth_pdb.benchmark import run_benchmark
    except ImportError as exc:
        logger.error("Failed to import benchmark: %s", exc)
        logger.error("Install requirements: pip install synth-pdb[gnn] transformers accelerate")
        return 1

    logger.info(
        "Starting benchmark: %d structures, predictor=%s", args.n_structures, args.predictor
    )

    results = run_benchmark(
        n_structures=args.n_structures,
        lengths=args.lengths,
        conformations=args.conformations,
        predictor=args.predictor,
        compute_shifts=not args.no_shifts,
        compute_gnn=not args.no_gnn,
        random_state=args.random_state,
    )

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + results.summary())

    # ── Save CSV ───────────────────────────────────────────────────────
    if args.output:
        results.to_csv(args.output)
        print(f"\nResults saved to: {args.output}")

    # Return 0 if at least 50% succeeded, 1 otherwise
    success_rate = results.n_success / results.n_structures if results.n_structures > 0 else 0
    return 0 if success_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
