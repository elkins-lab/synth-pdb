#!/usr/bin/env python3
"""
Scientific Benchmarking & Defensibility Suite (Expanded).

This script provides a comprehensive validation of synth-pdb against
experimental benchmarks, covering:
1. Chemical Shift Correlation (Pearson R, RMSD) - Supports SHIFTX2 and SPARTA+
2. Global Biophysics (Radius of Gyration)
3. Geometric Quality (Ramachandran, Rotamers, Z-scores)
4. Restraint Satisfaction (NOE violations)

SCIENTIFIC OBJECTIVE:
To prove the physical and biological defensibility of synthetic models
by matching ground-truth experimental data from the BMRB and PDB.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import biotite.structure.io.pdb as pdb_io

# Ensure the repo root is on sys.path regardless of the working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synth_pdb.bmrb_api import BMRBAPI
from synth_pdb.chemical_shifts import predict_chemical_shifts, calculate_shift_metrics
from synth_pdb.saxs import calculate_radius_of_gyration
from synth_pdb.validator import PDBValidator
from synth_pdb.quality.plots import (
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    save_publication_figure,
)

# Standard benchmarks
BENCHMARKS = [
    {"pdb": "1D3Z", "bmrb": "6457", "name": "Ubiquitin (1D3Z)"},
    {"pdb": "1UBQ", "bmrb": "17769", "name": "Ubiquitin (1UBQ)"},
]

# Manual offsets for BMRB vs PDB alignment
# (BMRB_res_id - PDB_res_id)
BMRB_OFFSETS = {
    "15430": 0,
    "6457": 0,
    "4022": 0,
}


def run_single_benchmark(pdb_id, bmrb_id, name, output_dir, use_shiftx2=True):
    """Run full suite for a single protein."""
    print("\n" + "=" * 60)
    print(f" BENCHMARK: {name} ({pdb_id} vs BMRB {bmrb_id})")
    print("=" * 60)

    # Resolve the PDB path relative to the repo root so the script works
    # correctly regardless of which directory it is invoked from.
    pdb_path = str(_REPO_ROOT / "examples" / f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"   [DOWNLOAD] Fetching {pdb_id} from RCSB...")
        BMRBAPI.download_pdb(pdb_id, pdb_path)

    # 1. Load Structure
    pdb_file = pdb_io.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)

    # Clean structure: keep only protein atoms (removes HOH, ligands)
    from biotite.structure import filter_amino_acids

    structure = structure[filter_amino_acids(structure)]

    # Re-save cleaned structure for validator
    temp_pdb = os.path.join(output_dir, f"{pdb_id}_clean.pdb")
    pdb_file.set_structure(structure)
    pdb_file.write(temp_pdb)
    with open(temp_pdb) as f:
        pdb_content = f.read()

    # 2. Geometric Validation
    print("   [GEOMETRY] Assessing physical quality...")
    validator = PDBValidator(pdb_content)
    # Fetch restraints from BMRB for satisfaction check
    restraints = BMRBAPI.fetch_restraints(bmrb_id)
    quality_report = validator.get_quality_report(nmr_restraints=restraints)

    # 3. Chemical Shifts
    print(f"   [NMR] Validating chemical shifts (SHIFTX2={use_shiftx2})...")
    exp_shifts_full = BMRBAPI.fetch_chemical_shifts(bmrb_id)
    # predict_chemical_shifts returns {chain: {res_id: {atom: val}}}
    syn_shifts_full = predict_chemical_shifts(structure, use_shiftx2=use_shiftx2)

    # Standardize on first chain
    first_chain = list(syn_shifts_full.keys())[0]
    syn_shifts = syn_shifts_full[first_chain]

    # Apply Offset if needed
    offset = BMRB_OFFSETS.get(bmrb_id, 0)

    nmr_results = {}
    target_atoms = ["CA", "CB", "N", "HA"]

    for atom in target_atoms:
        obs_vals = []
        pred_vals = []
        # Alignment logic
        for res_id, atoms in exp_shifts_full.items():
            pdb_res_id = int(res_id) - offset
            if atom in atoms and pdb_res_id in syn_shifts and atom in syn_shifts[pdb_res_id]:
                obs_vals.append(atoms[atom])
                pred_vals.append(syn_shifts[pdb_res_id][atom])

        if len(obs_vals) > 5:
            metrics = calculate_shift_metrics(np.array(obs_vals), np.array(pred_vals))
            nmr_results[atom] = {
                "pearson_r": float(metrics["correlation"]),
                "rmsd": float(metrics["rmsd"]),
                "n": len(obs_vals),
            }
            # Generate individual publication plot
            plot_chemical_shift_correlation(
                exp_shifts_full,
                syn_shifts,
                atom_type=atom,
                title=f"{name} {atom} Correlation",
                output_path=os.path.join(output_dir, f"{pdb_id}_{atom}_corr.pdf"),
            )

    # 4. Global Biophysics
    rg = calculate_radius_of_gyration(structure)

    # 5. Generate Suite Visualizations
    from biotite.structure import dihedral_backbone

    phi, psi, _ = dihedral_backbone(structure)
    mask = ~np.isnan(phi) & ~np.isnan(psi)
    plot_ramachandran_publication(
        np.degrees(phi[mask]),
        np.degrees(psi[mask]),
        title=f"Ramachandran: {name}",
        output_path=os.path.join(output_dir, f"{pdb_id}_ramachandran.pdf"),
    )

    # Summary Result
    result = {
        "protein": name,
        "pdb": pdb_id,
        "bmrb": bmrb_id,
        "biophysics": {"rg": float(rg)},
        "geometry": quality_report,
        "nmr_correlation": nmr_results,
        "defensible": quality_report["is_overall_scientifically_defensible"],
    }

    print(f"\n   [SUMMARY] Scientific Report for {name}:")
    print(f"      - Mean CA Correlation: {nmr_results.get('CA', {}).get('pearson_r', 0.0):.3f}")
    print(f"      - Radius of Gyration:  {rg:.2f} A")
    print(f"      - Potential Energy:    {quality_report['potential_energy_kj_mol']:.2e} kJ/mol")
    print(
        f"      - Ramachandran Favored: {quality_report['ramachandran_stats']['favored_pct']:.1f}%"
    )
    print(
        f"      - Bond Z-score (mean): {quality_report['geometric_z_scores']['mean_bond_zscore']:.2f}"
    )

    if quality_report.get("total_restraints"):
        print(f"      - NOE Satisfaction:   {quality_report.get('noe_satisfaction_pct', 0.0):.1f}%")

    if not result["defensible"]:
        print("      - Defensibility Failures:")
        z = quality_report["geometric_z_scores"]
        r = quality_report["ramachandran_stats"]
        rot = quality_report["rotamer_stats"]
        e = quality_report["potential_energy_kj_mol"]
        s = quality_report["hydrophobic_burial_ratio"]

        if z["mean_bond_zscore"] >= 3.0:
            print(f"        * Bond Z-score too high ({z['mean_bond_zscore']:.2f})")
        if rot["favored_rotamers_pct"] <= 80.0:
            print(f"        * Rotamers below threshold ({rot['favored_rotamers_pct']:.1f}%)")
        if r["outlier_pct"] >= 5.0:
            print(f"        * Ramachandran outliers too high ({r['outlier_pct']:.1f}%)")
        if e >= 1e5:
            print(f"        * Potential energy too high ({e:.2e})")
        if s < 0.8:
            print(f"        * Burial ratio too low ({s:.2f})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Expanded Scientific Benchmarking Suite")
    parser.add_argument("--full", action="store_true", help="Run against all standard benchmarks")
    parser.add_argument("--pdb", default="1D3Z", help="Single PDB ID")
    parser.add_argument("--bmrb", default="6457", help="Single BMRB ID")
    parser.add_argument(
        "--no-shiftx2", action="store_true", help="Force SPARTA+ (empirical) instead of SHIFTX2"
    )
    parser.add_argument("--output", default="artifacts/benchmarks", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    targets = (
        BENCHMARKS if args.full else [{"pdb": args.pdb, "bmrb": args.bmrb, "name": "User-Defined"}]
    )

    all_results = []
    for target in targets:
        try:
            res = run_single_benchmark(
                target["pdb"],
                target["bmrb"],
                target["name"],
                args.output,
                use_shiftx2=not args.no_shiftx2,
            )
            all_results.append(res)
        except Exception as e:
            print(f"   [ERROR] Failed to process {target['name']}: {e}")

    # Export JSON Report
    report_path = os.path.join(args.output, "defensibility_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] Scientific Defensibility Report exported to {report_path}")


if __name__ == "__main__":
    main()
