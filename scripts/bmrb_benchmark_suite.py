import logging
import os
from typing import List, Tuple

import biotite.structure.io.pdb as pdb_io
import numpy as np

from synth_pdb.bmrb_api import BMRBAPI
from synth_pdb.chemical_shifts import calculate_shift_metrics, predict_chemical_shifts

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# List of benchmark pairs: (BMRB_ID, PDB_ID, Protein_Name)
BENCHMARK_PAIRS = [
    ("6457", "1D3Z", "Ubiquitin"),
    ("17769", "1UBQ", "Ubiquitin"),
    ("4022", "1HDN", "HPr"),
    ("4141", "1GVP", "Gene 5 protein"),
    ("36464", "2LGI", "Protein G"),
    ("15430", "1PGB", "Protein G"),
    ("4216", "1CYO", "Cytochrome c"),
    ("4364", "1RFA", "A-tract DNA binding protein"),
]


def run_large_scale_benchmark(cache_dir: str = "artifacts/benchmark_cache") -> None:
    """Run chemical shift benchmarks across multiple BMRB entries."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print("\n" + "=" * 80)
    print(f"{'BMRB ID':<10} | {'PDB ID':<8} | {'Protein':<25} | {'CA Corr':<8} | {'CA RMSD':<8}")
    print("-" * 80)

    summary_results: List[Tuple[float, float]] = []

    for bmrb_id, pdb_id, name in BENCHMARK_PAIRS:
        pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")

        # 1. Download PDB if not in cache
        if not os.path.exists(pdb_path):
            success = BMRBAPI.download_pdb(pdb_id, pdb_path)
            if not success:
                logger.error(f"Failed to download PDB {pdb_id}. Skipping.")
                continue

        # 2. Fetch experimental shifts
        obs_shifts = BMRBAPI.fetch_chemical_shifts(bmrb_id)
        if not obs_shifts:
            logger.error(f"Failed to fetch shifts for BMRB {bmrb_id}. Skipping.")
            continue

        # 3. Load PDB and Predict
        try:
            pdb_file = pdb_io.PDBFile.read(pdb_path)
            structure = pdb_file.get_structure(model=1)

            # Predict
            predicted = predict_chemical_shifts(structure, use_shiftx2=False)
            first_chain = list(predicted.keys())[0]
            pred_shifts = predicted[first_chain]

            # 4. Align and Calculate Metrics for CA
            obs_vals = []
            pred_vals = []
            for res_id, atoms in obs_shifts.items():
                if "CA" in atoms and res_id in pred_shifts and "CA" in pred_shifts[res_id]:
                    obs_vals.append(atoms["CA"])
                    pred_vals.append(pred_shifts[res_id]["CA"])

            if len(obs_vals) > 10:
                metrics = calculate_shift_metrics(np.array(obs_vals), np.array(pred_vals))
                corr = metrics["correlation"]
                rmsd = metrics["rmsd"]

                print(f"{bmrb_id:<10} | {pdb_id:<8} | {name:<25} | {corr:>8.3f} | {rmsd:>8.3f}")
                summary_results.append((corr, rmsd))
            else:
                print(f"{bmrb_id:<10} | {pdb_id:<8} | {name:<25} | {'INSUFFICIENT DATA':<18}")

        except Exception as e:
            logger.error(f"Error processing {bmrb_id}/{pdb_id}: {e}")
            continue

    if summary_results:
        avg_corr = np.mean([r[0] for r in summary_results])
        avg_rmsd = np.mean([r[1] for r in summary_results])
        print("-" * 80)
        print(f"{'OVERALL AVERAGE':<47} | {avg_corr:>8.3f} | {avg_rmsd:>8.3f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_large_scale_benchmark()
