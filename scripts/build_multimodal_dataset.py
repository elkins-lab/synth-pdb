#!/usr/bin/env python3
"""
High-Performance Multi-Modal Dataset Builder for Protein AI.

Generates synchronized 3D structures, Cryo-EM maps, SAXS profiles, and NMR
observables in a single high-throughput pipeline.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure local synth_pdb is prioritized
current_path = Path(__file__).resolve().parent
repo_root = current_path.parent
if (repo_root / "synth_pdb").exists():
    sys.path.insert(0, str(repo_root))

from synth_pdb.batch_generator import BatchedGenerator  # noqa: E402
from synth_pdb.cryo_em import generate_density_map, save_mrc_file  # noqa: E402
from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier  # noqa: E402
from synth_pdb.rdc import calculate_rdcs  # noqa: E402
from synth_pdb.saxs import calculate_saxs_profile, export_saxs_profile  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multimodal_builder")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-modal synthetic protein dataset.")
    parser.add_argument(
        "--n", type=int, default=100, help="Number of structures to generate. Default 100."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        help="Amino acid sequence (1-letter codes). Default is Ubiquitin.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multimodal_dataset",
        help="Directory to save the dataset. Default 'multimodal_dataset'.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=3.5,
        help="Cryo-EM map resolution in Angstroms. Default 3.5Å.",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=0.5,
        help="Maximum SAXS q-vector (A^-1). Default 0.5.",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=5.0,
        help="Conformational drift (degrees) for ensemble diversity. Default 5.0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--skip-mrc", action="store_true", help="Skip Cryo-EM map generation.")
    parser.add_argument("--skip-saxs", action="store_true", help="Skip SAXS profile generation.")
    parser.add_argument("--skip-nmr", action="store_true", help="Skip NMR observable generation.")
    parser.add_argument(
        "--quality-model",
        type=str,
        default=None,
        help="Path to GNN quality model (.pt) to filter structures.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum P(Good) probability to keep a structure. Default 0.7.",
    )

    args = parser.parse_args()

    # Create directory structure
    out_path = Path(args.output_dir)
    dirs = {
        "pdb": out_path / "pdbs",
        "mrc": out_path / "mrcs",
        "saxs": out_path / "saxs",
        "nmr": out_path / "nmr",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Multi-Modal Dataset Build (N={args.n})...")
    start_time = time.time()

    # 1. Generate Ensemble (Vectorized)
    # Full atom mode is required for SAXS and Cryo-EM fidelity
    logger.info(f"Step 1: Generating {args.n} structures using BatchedGenerator...")
    bg = BatchedGenerator(args.sequence, n_batch=args.n, full_atom=True)
    batch = bg.generate_batch(drift=args.drift, seed=args.seed)

    # ── Quality Filtering ───────────────────────────────────────────
    if args.quality_model:
        logger.info(
            f"Applying quality filter using {args.quality_model} (threshold={args.min_quality})..."
        )
        clf = GNNQualityClassifier(args.quality_model)
        scores = clf.score_batch(batch)

        # Create boolean mask for 'Good' structures
        mask = np.array([s.global_score >= args.min_quality for s in scores])
        n_kept = int(np.sum(mask))
        logger.info(f"Quality Filter: Kept {n_kept}/{args.n} structures ({n_kept / args.n:.1%}).")

        if n_kept == 0:
            logger.error("No structures passed the quality filter. Aborting.")
            return

        # Filter the BatchedPeptide object
        batch = batch[mask]
        # Update n for the loop below
        args.n = n_kept

    # Pre-convert to Biotite Stack for simulation functions
    stack = batch.to_stack()
    logger.info(f"Ensemble ready. Structure shape: {batch.coords.shape}")

    metadata = []

    # 2. Iterate and Simulate Modalities
    for i in range(args.n):
        sample_id = f"sample_{i:04d}"
        if i % 10 == 0:
            logger.info(f"Processing {sample_id}...")

        # A. Save PDB
        pdb_path = dirs["pdb"] / f"{sample_id}.pdb"
        batch.save_pdb(str(pdb_path), index=i)

        row: dict[str, Any] = {"id": sample_id, "pdb_path": str(pdb_path)}

        # B. Cryo-EM (Individual Map per sample)
        if not args.skip_mrc:
            mrc_path = dirs["mrc"] / f"{sample_id}.mrc"
            density, origin = generate_density_map(stack[i], resolution=args.resolution)
            save_mrc_file(str(mrc_path), density, origin)
            row["mrc_path"] = str(mrc_path)

        # C. SAXS
        if not args.skip_saxs:
            saxs_path = dirs["saxs"] / f"{sample_id}.dat"
            q, intensity = calculate_saxs_profile(stack[i], q_max=args.q_max)
            export_saxs_profile(q, intensity, str(saxs_path))
            row["saxs_path"] = str(saxs_path)

        # D. NMR (RDCs)
        if not args.skip_nmr:
            nmr_path = dirs["nmr"] / f"{sample_id}_rdc.csv"
            # Calculate RDCs with standard alignment tensor (Da=10, R=0.15)
            rdcs = calculate_rdcs(stack[i], da=10.0, r=0.15)

            with open(nmr_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["residue", "rdc_hz"])
                for res_id, val in sorted(rdcs.items()):
                    writer.writerow([res_id, f"{val:.4f}"])
            row["nmr_path"] = str(nmr_path)

        metadata.append(row)

    # 3. Save Master Metadata
    csv_path = out_path / "metadata.csv"
    if metadata:
        with open(csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
            dict_writer.writeheader()
            dict_writer.writerows(metadata)

    elapsed = time.time() - start_time
    logger.info(f"Successfully built dataset at {out_path.absolute()}")
    logger.info(f"Total Time: {elapsed:.2f}s ({elapsed / args.n:.3f} s/sample)")


if __name__ == "__main__":
    main()
