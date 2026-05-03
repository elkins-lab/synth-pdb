#!/usr/bin/env python3
import time
import sys
import os

sys.path.insert(0, os.getcwd())
import numpy as np
from synth_pdb.validator import PDBValidator
from synth_pdb.generator import generate_pdb_content
import logging

# Disable logging for cleaner output
logging.getLogger("synth_pdb").setLevel(logging.ERROR)


def benchmark_steric_tweak(n_residues=100, conformation="alpha"):
    print(f"\n--- Benchmarking Steric Clash Tweak ({n_residues} residues, {conformation}) ---")

    # 1. Generate a large structure
    # Use alpha with some drift to simulate realistic refinement
    pdb_content = generate_pdb_content(length=n_residues, conformation=conformation, drift=2.0)
    atoms = PDBValidator._parse_pdb_atoms(pdb_content)
    num_atoms = len(atoms)
    print(f"Atoms: {num_atoms}")

    # 2. Original Logic (Reconstructed from committed code)
    # We include the overhead that was actually there: function calls, lookups
    def original_logic(parsed_atoms):
        from synth_pdb.validator import VAN_DER_WAALS_RADII

        modified_atoms = [atom.copy() for atom in parsed_atoms]
        num = len(modified_atoms)

        # Bonded pairs construction (was inside the function)
        temp_grouped_atoms = {}
        for atom in modified_atoms:
            c, r, n = atom["chain_id"], atom["residue_number"], atom["atom_name"]
            if c not in temp_grouped_atoms:
                temp_grouped_atoms[c] = {}
            if r not in temp_grouped_atoms[c]:
                temp_grouped_atoms[c][r] = {}
            temp_grouped_atoms[c][r][n] = atom

        bonded_pairs = set()
        for _c, res_dict in temp_grouped_atoms.items():
            sorted_res = sorted(res_dict.keys())
            for i, r_num in enumerate(sorted_res):
                r = res_dict[r_num]
                n, ca, c, o = r.get("N"), r.get("CA"), r.get("C"), r.get("O")
                if n and ca:
                    bonded_pairs.add(tuple(sorted((n["atom_number"], ca["atom_number"]))))
                if ca and c:
                    bonded_pairs.add(tuple(sorted((ca["atom_number"], c["atom_number"]))))
                if c and o:
                    bonded_pairs.add(tuple(sorted((c["atom_number"], o["atom_number"]))))
                if i + 1 < len(sorted_res):
                    next_n = res_dict[sorted_res[i + 1]].get("N")
                    if c and next_n:
                        bonded_pairs.add(tuple(sorted((c["atom_number"], next_n["atom_number"]))))

        clashes = 0
        for i in range(num):
            a1 = modified_atoms[i]
            for j in range(i + 1, num):
                a2 = modified_atoms[j]
                if a1["atom_number"] == a2["atom_number"]:
                    continue
                if tuple(sorted((a1["atom_number"], a2["atom_number"]))) in bonded_pairs:
                    continue

                dist = np.linalg.norm(a1["coords"] - a2["coords"])
                vdw1 = VAN_DER_WAALS_RADII.get(a1["element"], 1.5)
                vdw2 = VAN_DER_WAALS_RADII.get(a2["element"], 1.5)
                if dist < 2.0 or dist < (vdw1 + vdw2) * 0.8:
                    clashes += 1
        return clashes

    start_old = time.time()
    _ = original_logic(atoms)
    end_old = time.time()
    time_old = end_old - start_old
    print(f"Original Python Logic: {time_old:.4f}s")

    # 3. Vectorized Logic (Current)
    start_new = time.time()
    _ = PDBValidator._apply_steric_clash_tweak(atoms)
    end_new = time.time()
    time_new = end_new - start_new
    print(f"Vectorized Logic:      {time_new:.4f}s")

    speedup = time_old / time_new if time_new > 0 else float("inf")
    print(f"Speedup: {speedup:.1f}x")


def benchmark_string_overhead(n_iterations=50):
    print(f"\n--- Benchmarking Refinement Overhead ({n_iterations} iterations) ---")
    n_residues = 50
    pdb_content = generate_pdb_content(length=n_residues, conformation="random")
    atoms = PDBValidator._parse_pdb_atoms(pdb_content)

    # A. Measure Assembly + Parsing (The old way)
    start_str = time.time()
    for _ in range(n_iterations):
        # Assemble
        _ = PDBValidator.atoms_to_pdb_content(atoms)
        # Parse
        _ = PDBValidator._parse_pdb_atoms(pdb_content)
    end_str = time.time()
    time_str = end_str - start_str
    print(f"Time spent in String assembly/parsing: {time_str:.4f}s")

    # B. Measure Direct Object Initialization (The new way)
    start_obj = time.time()
    for _ in range(n_iterations):
        # Just init the validator with atoms
        _ = PDBValidator(parsed_atoms=atoms)
    end_obj = time.time()
    time_obj = end_obj - start_obj
    print(f"Time spent in direct Object init:      {time_obj:.4f}s")

    reduction = (1 - (time_obj / time_str)) * 100 if time_str > 0 else 0
    print(f"Overhead Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    # Test on a realistic large protein (e.g. 150 residues)
    benchmark_steric_tweak(150, "alpha")
    # Test on a small peptide
    benchmark_steric_tweak(30, "alpha")
    # Refinement loop overhead
    benchmark_string_overhead(100)
