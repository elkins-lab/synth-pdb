import os

import biotite.sequence as seq
import biotite.structure.io.pdb as pdb_io
import numpy as np
import pynmrstar
import pytest

from synth_pdb.chemical_shifts import calculate_shift_metrics, predict_chemical_shifts

# Skip if pynmrstar is not available
pytest.importorskip("pynmrstar")


def fetch_bmrb_shifts(bmrb_id: str) -> tuple[dict[int, dict[str, float]], str]:
    """Fetch experimental shifts from BMRB and return as nested dict and sequence."""
    try:
        entry = pynmrstar.Entry.from_database(bmrb_id)
    except Exception as e:
        pytest.skip(f"Failed to fetch BMRB {bmrb_id}: {e}")
        return {}, ""

    shift_loops = entry.get_loops_by_category("_Atom_chem_shift")
    if not shift_loops:
        return {}, ""

    loop = shift_loops[0]
    tag_to_idx = {tag.lower(): i for i, tag in enumerate(loop.tags)}

    idx_res = tag_to_idx.get("comp_index_id") or tag_to_idx.get("seq_id")
    idx_atom = tag_to_idx.get("atom_id")
    idx_val = tag_to_idx.get("val")
    idx_comp = tag_to_idx.get("comp_id")

    shifts: dict[int, dict[str, float]] = {}
    sequence_map: dict[int, str] = {}
    for row in loop.data:
        try:
            res_id = int(row[idx_res])
            atom_name = row[idx_atom]
            val = float(row[idx_val])
            comp_id = row[idx_comp]

            # Standardize atom names
            if atom_name == "H":
                atom_name = "HN"

            if res_id not in shifts:
                shifts[res_id] = {}
                sequence_map[res_id] = comp_id
            shifts[res_id][atom_name] = val
        except (ValueError, TypeError):
            continue

    # Construct sequence string
    sorted_res = sorted(sequence_map.keys())
    sequence_list = []
    for r in sorted_res:
        try:
            sequence_list.append(seq.ProteinSequence.convert_letter_3to1(sequence_map[r]))
        except KeyError:
            sequence_list.append("X")
    sequence = "".join(sequence_list)

    return shifts, sequence


class TestBMRBBenchmark:
    """
    Experimental Validation Benchmark Suite.
    Compares synth-pdb predictions against high-resolution BMRB datasets.
    """

    @pytest.mark.parametrize(
        "bmrb_id, pdb_path, expected_ca_r, expected_ca_rmsd",
        [
            ("6457", "examples/1D3Z.pdb", 0.95, 1.30),
            ("17769", "examples/1UBQ.pdb", 0.95, 1.30),
            ("6457", "examples/1UBQ.pdb", 0.95, 1.30),
        ],
    )
    def test_shift_accuracy_benchmark(
        self, bmrb_id: str, pdb_path: str, expected_ca_r: float, expected_ca_rmsd: float
    ) -> None:
        """Verify predicted shifts against experimental BMRB data."""
        if not os.path.exists(pdb_path):
            pytest.skip(f"PDB file {pdb_path} not found.")

        # 1. Load PDB
        pdb_file = pdb_io.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1)

        # 2. Predict Shifts
        predicted = predict_chemical_shifts(structure, use_shiftx2=False)
        first_chain = list(predicted.keys())[0]
        pred_shifts = predicted[first_chain]

        # 3. Fetch Experimental Shifts
        obs_shifts, bmrb_seq = fetch_bmrb_shifts(bmrb_id)
        assert len(obs_shifts) > 0, f"No shifts found for BMRB {bmrb_id}"

        # 4. Verify Sequence Match (Heuristic: at least 90% identity)
        # Get PDB sequence
        pdb_seq = "".join(
            seq.ProteinSequence.convert_letter_3to1(r)
            for r in structure[structure.atom_name == "CA"].res_name
        )

        # Simple overlap check
        min_len = min(len(bmrb_seq), len(pdb_seq))
        matches = sum(1 for i in range(min_len) if bmrb_seq[i] == pdb_seq[i])
        identity = matches / min_len if min_len > 0 else 0

        print(f"\nComparing BMRB {bmrb_id} to {pdb_path}:")
        print(f"  Sequence Identity: {identity:.1%}")

        if identity < 0.9:
            pytest.skip(
                f"Sequence mismatch between BMRB {bmrb_id} and PDB {pdb_path} ({identity:.1%})"
            )

        # 5. Align and Compare for specific atom types
        atom_types = ["CA", "CB", "HA", "HN", "N"]

        results = {}
        for atom in atom_types:
            obs_vals = []
            pred_vals = []

            for res_id, atoms in obs_shifts.items():
                # Note: BMRB and PDB might have different offsets, but usually Ubiquitin is 1-indexed in both
                if atom in atoms and res_id in pred_shifts and atom in pred_shifts[res_id]:
                    obs_vals.append(atoms[atom])
                    pred_vals.append(pred_shifts[res_id][atom])

            if len(obs_vals) > 10:
                metrics = calculate_shift_metrics(np.array(obs_vals), np.array(pred_vals))
                results[atom] = metrics
                print(
                    f"  {atom} Metrics (n={len(obs_vals)}): RMSD={metrics['rmsd']:.3f}, Corr={metrics['correlation']:.3f}"
                )

        # 6. Scientific Assertions
        assert "CA" in results
        assert results["CA"]["correlation"] >= expected_ca_r
        assert results["CA"]["rmsd"] <= expected_ca_rmsd

        # Ensure heavy atoms (CA, CB) generally perform well
        assert results["CB"]["correlation"] > 0.9
