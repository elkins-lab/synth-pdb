from typing import Any, Dict, Tuple
import os

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import pytest
import numpy as np

from synth_pdb.geometry.rmsd import calculate_rmsd
from synth_pdb.geometry.superposition import find_medoid, superimpose_structures

# =============================================================================
# SCIENTIFIC VALIDATION SUITE: GEOMETRY MODULE
# =============================================================================
#
# This suite validates the synth-pdb geometry kernels (RMSD, Kabsch superposition,
# and Dihedral calculations) against peer-reviewed structural biology benchmarks.
#
# EDUCATIONAL PRINCIPLES:
# 1. Representative Model (Medoid): When comparing NMR ensembles to single X-ray
#    crystal structures, we select the "medoid" rather than the "centroid"
#    (average coordinates).
#    - SCIENTIFIC RATIONALE: The medoid is an actual member of the ensemble
#      with physical bond lengths and angles. A mathematical average (centroid)
#      often results in non-physical "ghost" atoms where bond lengths are
#      distorted in regions of high mobility.
#
# 2. Structural Mapping: We use optimal sequence alignment (Needleman-Wunsch)
#    to pair atoms even when PDB files have different numbering or expression
#    tags (e.g., N-terminal purification tags).
#
# 3. Sliding Window Analysis: For proteins with flexible loops or linkers,
#    a global RMSD can be misleading. We find the "best-fit" window to
#    validate that the local geometry kernels correctly identify structural
#    correspondence where it exists.
#    - APPLICATION: This is essential for the NESG benchmarks where the NMR
#      construct may include disordered His-tags not seen in the crystal.
#
# REFERENCES:
# - Ubiquitin: Cornilescu et al. (1998) J. Am. Chem. Soc. 120, 6836-6837.
#   DOI: 10.1021/ja9812610 (NMR: 1D3Z, X-ray: 1UBQ)
# - BPTI: Otting & Wuthrich (1989) J. Am. Chem. Soc. 111, 1871-1875.
#   DOI: 10.1021/ja00187a042 (NMR: 1PIT, X-ray: 5PTI)
# - NESG CtR107 (Montelione Group): Everett et al. (2009) J. Biomol. NMR 45, 13–21.
#   DOI: 10.1007/s10858-009-9333-x (NMR: 2KCU, X-ray: 3E0H)
#   - NOTE: CtR107 is a GyrI-like protein from Chlorobaculum tepidum. This pair
#     was used by the NESG to demonstrate that NMR and X-ray structures can
#     converge to high accuracy (< 1.2 Å) when high-quality NMR data is used.
# =============================================================================


def get_best_window_rmsd(
    nmr_model: struc.AtomArray, xray_struct: struc.AtomArray, window_size: int = 30
) -> float:
    """
    Finds the structurally best-matching window between two models.

    EDUCATIONAL NOTE: This sliding-window approach (Local RMSD) is used in
    structural biology to find "rigid bodies" within larger, flexible proteins.
    It ensures that a small conformational change (like a hinge movement)
    doesn't obscure the high accuracy of the individual domain representations.
    """

    def get_bk(m: struc.AtomArray) -> Tuple[struc.AtomArray, np.ndarray]:
        # Filter for standard amino acids and backbone atoms (N, CA, C)
        # We exclude Oxygens as they are more sensitive to refinement noise.
        m = m[struc.filter_amino_acids(m)]
        bk = m[(m.atom_name == "N") | (m.atom_name == "CA") | (m.atom_name == "C")]
        return bk, struc.get_residue_starts(bk)

    n_bk, n_starts = get_bk(nmr_model)
    best_overall_rmsd = float("inf")

    # Iterate through all chains in the X-ray structure (handling oligomers)
    for chain_id in struc.get_chains(xray_struct):
        x_bk, x_starts = get_bk(xray_struct[xray_struct.chain_id == chain_id])
        if len(x_starts) < window_size:
            continue

        # Sliding window structural comparison
        for i in range(len(n_starts) - window_size):
            n_slice = n_bk[n_starts[i] : n_starts[i + window_size]]
            cn = n_slice.coord

            for j in range(len(x_starts) - window_size):
                x_slice = x_bk[x_starts[j] : x_starts[j + window_size]]
                # Ensure we are comparing equal-length segments
                if len(x_slice) != len(n_slice):
                    continue

                cx = x_slice.coord
                # Calculate the minimal RMSD via Kabsch superposition
                # aligned = (R @ cn.T).T + t
                aligned = superimpose_structures(cn, cx)
                rmsd = calculate_rmsd(aligned, cx)
                if rmsd < best_overall_rmsd:
                    best_overall_rmsd = rmsd

    return best_overall_rmsd


@pytest.mark.parametrize(
    "pair",
    [
        {
            "name": "Ubiquitin (Bax et al.)",
            "nmr": "1D3Z.pdb",
            "xray": "1UBQ.pdb",
            "target": 0.6,
            "win": 50,
            "citation": "JACS 1998, 120:6836",
            "description": "Gold-standard comparison for small proteins.",
        },
        {
            "name": "BPTI (Wuthrich et al.)",
            "nmr": "1PIT.pdb",
            "xray": "5PTI.pdb",
            "target": 1.2,
            "win": 50,
            "citation": "JACS 1989, 111:1871",
            "description": "Canonical benchmark for NMR/X-ray differences in loops.",
        },
        {
            "name": "NESG CtR107 (Montelione et al.)",
            "nmr": "2KCU.pdb",
            "xray": "3E0H.pdb",
            "target": 1.5,
            "win": 30,
            "citation": "J. Biol. NMR 2009, 45:13",
            "description": "NESG demonstration of structural convergence for a 160-res domain.",
        },
    ],
)
def test_scientific_validation_benchmarks(pair: Dict[str, Any]) -> None:
    """
    Validate geometry module against canonical structural biology targets.

    This test ensures that the synth-pdb geometry kernels can reproduce
    high-quality structural alignments cited in peer-reviewed literature.
    """
    # Resolve PDB paths relative to this test file (moving them from root to examples/)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(test_dir, "..", "examples")

    nmr_path = os.path.join(examples_dir, pair["nmr"])
    xr_path = os.path.join(examples_dir, pair["xray"])

    if not os.path.exists(nmr_path):
        pytest.skip(f"File {pair['nmr']} not found in examples/")
    nmr_stk = pdb.get_structure(pdb.PDBFile.read(nmr_path))
    xr_struct = pdb.get_structure(pdb.PDBFile.read(xr_path), model=1)

    # 1. Identify the medoid model of the NMR ensemble for comparison
    # NMR structures are deposited as ensembles (usually 20 models).
    # We select the "Medoid" - the structure with the minimal distance to all others.
    models = nmr_stk[0:20]
    msk = (models.atom_name == "CA") & struc.filter_amino_acids(models)

    # Extract CA coordinates for all models to find the medoid
    coords_list = [m.coord for m in models[:, msk]]
    idx = find_medoid(coords_list)
    nmr_medoid = models[idx]

    # 2. Find the best structural correspondence using regional matching
    # This sliding-window RMSD accounts for different expression constructs.
    rmsd = get_best_window_rmsd(nmr_medoid, xr_struct, window_size=pair["win"])

    print(f"\nBenchmark: {pair['name']}")
    print(f"Reference: {pair['citation']}")
    print(f"Notes: {pair['description']}")
    print(f"Result (Best {pair['win']}-res window): {rmsd:.3f} A")

    # Assert result is within physics-informed thresholds
    # Values under 1.5A are generally considered 'identical' folds in structural biology.
    assert rmsd < pair["target"]
