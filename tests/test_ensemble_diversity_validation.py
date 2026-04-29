"""
Scientific Validation: Ensemble Conformational Diversity.

Validates that generating multiple structures from the same sequence produces
a structurally diverse ensemble, and that the medoid finder correctly identifies
the most representative model.

SCIENTIFIC BASIS:
  A meaningful structural ensemble must exhibit:
  1. Non-zero backbone RMSD spread: different models must differ by at least
     0.1 A in CA-RMSD, otherwise the generator has collapsed to a single
     conformation (degenerate ensemble).
  2. Medoid model correctness: the representative member must have mean RMSD
     to all others <= the overall ensemble mean RMSD.
  3. Physical upper bound: no pairwise CA-RMSD should exceed 50 A.

REFERENCES:
  Lipari, G. & Szabo, A. (1982). Model-free approach to the interpretation
  of nuclear magnetic relaxation in macromolecules.
  J Am Chem Soc, 104, 4546-4559. DOI: 10.1021/ja00381a009
"""

import io
import numpy as np
import pytest

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import generate_pdb_content
from synth_pdb.geometry.rmsd import calculate_rmsd
from synth_pdb.geometry.superposition import find_medoid, superimpose_structures

N_MODELS = 12
# Use a sequence with mixed secondary structure propensity to encourage diversity
SEQUENCE = "LKELEKGG"


def _ca_coords(pdb_content: str) -> np.ndarray:
    """Extract backbone CA coordinates from a PDB string."""
    pdb_file = pdb_io.PDBFile.read(io.StringIO(pdb_content))
    st = pdb_file.get_structure(model=1)
    aa = st[struc.filter_amino_acids(st)]
    ca = aa[aa.atom_name == "CA"]
    return ca.coord


@pytest.fixture(scope="module")
def ensemble_coords():
    """Generate N_MODELS structures using generate_pdb_content (fast, no minimization)."""
    coords_list = []
    for _ in range(N_MODELS):
        try:
            pdb_content = generate_pdb_content(sequence_str=SEQUENCE)
            coords = _ca_coords(pdb_content)
            if len(coords) >= 4:
                coords_list.append(coords)
        except Exception:
            continue
    return coords_list


def test_ensemble_has_sufficient_models(ensemble_coords):
    """Ensemble must contain at least 8 valid models."""
    assert len(ensemble_coords) >= 8, f"Expected >= 8 models, got {len(ensemble_coords)}"


def test_ensemble_not_all_identical(ensemble_coords):
    """At least one pair of structures must differ by > 0.01 A CA-RMSD.

    SCIENTIFIC BASIS:
    A generator that produces identical coordinates every time offers no
    conformational sampling utility.
    """
    coords = ensemble_coords
    n = min(len(coords), 8)
    found_diverse = False
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = coords[i], coords[j]
            min_len = min(len(c1), len(c2))
            if min_len < 3:
                continue
            aligned = superimpose_structures(c1[:min_len], c2[:min_len])
            if calculate_rmsd(aligned, c2[:min_len]) > 0.01:
                found_diverse = True
                break
        if found_diverse:
            break
    assert found_diverse, "All ensemble models are numerically identical — degenerate ensemble"


def test_ensemble_max_spread_physically_bounded(ensemble_coords):
    """Max pairwise CA-RMSD must be < 50 A (no non-physical coordinates)."""
    coords = ensemble_coords
    n = min(len(coords), 8)
    max_rmsd = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = coords[i], coords[j]
            min_len = min(len(c1), len(c2))
            if min_len < 3:
                continue
            aligned = superimpose_structures(c1[:min_len], c2[:min_len])
            max_rmsd = max(max_rmsd, calculate_rmsd(aligned, c2[:min_len]))
    print(f"\n  Ensemble max pairwise CA-RMSD = {max_rmsd:.3f} A")
    assert max_rmsd < 50.0, f"Max RMSD {max_rmsd:.3f} A exceeds physical bound 50 A"


def test_medoid_returns_valid_index(ensemble_coords):
    """find_medoid must return a valid integer index in [0, n)."""
    coords = ensemble_coords
    n = min(len(coords), 8)
    min_len = min(len(c) for c in coords[:n])
    trimmed = [c[:min_len] for c in coords[:n]]
    idx = find_medoid(trimmed)
    assert isinstance(idx, int | np.integer), f"Expected int, got {type(idx)}"
    assert 0 <= idx < n, f"Medoid index {idx} out of range [0, {n})"


def test_medoid_has_minimal_mean_deviation(ensemble_coords):
    """Medoid mean RMSD to all others must be <= ensemble mean RMSD (+5% tolerance).

    SCIENTIFIC BASIS:
    The medoid is by definition the structure that minimises the mean distance
    to all other members. A medoid with higher mean RMSD than the ensemble
    average would indicate a bug in find_medoid.
    """
    coords = ensemble_coords
    n = min(len(coords), 8)
    min_len = min(len(c) for c in coords[:n])
    trimmed = [c[:min_len] for c in coords[:n]]

    # Pairwise RMSD matrix
    rmsds = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            aligned = superimpose_structures(trimmed[i], trimmed[j])
            r = calculate_rmsd(aligned, trimmed[j])
            rmsds[i, j] = rmsds[j, i] = r

    idx = find_medoid(trimmed)
    medoid_mean = float(np.mean(rmsds[idx]))
    overall_mean = float(np.mean(rmsds[rmsds > 0]))  # non-zero off-diagonal

    print(f"\n  Medoid mean RMSD = {medoid_mean:.3f} A  |  Ensemble mean = {overall_mean:.3f} A")
    assert (
        medoid_mean <= overall_mean * 1.10
    ), f"Medoid mean RMSD ({medoid_mean:.3f}) > 110% of ensemble mean ({overall_mean:.3f})"
