import biotite.structure as struc
import pytest

from synth_pdb.nmr import calculate_rpf_score


def test_rpf_perfect_fit() -> None:
    """If all restraints are satisfied and no extra short distances exist, RPF should be 1.0."""
    # Create a simple structure: two protons at 3.0 A
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    # Create a restraint for them
    restraints = [{"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}]

    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["f_measure"] == 1.0


def test_rpf_unsatisfied_restraint() -> None:
    """If a restraint is violated (distance > upper bound), recall should decrease."""
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([6, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    # Restraint says < 5.0, but it's 6.0
    restraints = [{"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}]

    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 0.0
    assert scores["f_measure"] == 0.0


def test_rpf_low_precision() -> None:
    """If the structure has many short distances not in the restraints, precision should be low."""
    # Three protons in a triangle, all at 3.0 A
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    h3 = struc.Atom([0, 3, 0], atom_name="H", element="H", res_id=3, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2, h3])

    # Only one restraint (H1-H2)
    restraints = [{"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}]

    # Structure has 3 short distances: (1,2), (1,3), (2,3)
    # Only (1,2) is in restraints.
    # Precision = 1/3 = 0.33
    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(0.333, abs=0.01)
    # F = 2 * (1 * 1/3) / (1 + 1/3) = (2/3) / (4/3) = 0.5
    assert scores["f_measure"] == pytest.approx(0.5, abs=0.01)


def test_rpf_empty_restraints() -> None:
    """Empty restraints should return zeros."""
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    structure = struc.array([h1])
    scores = calculate_rpf_score(structure, [])
    assert scores == {"recall": 0.0, "precision": 0.0, "f_measure": 0.0}


def test_rpf_sparse_protons() -> None:
    """If the structure has fewer than 2 protons, precision should be 1.0 (no false positives possible).

    SCIENTIFIC BASIS:
    Precision in RPF measures the fraction of model contacts supported by restraints.
    If there are no model contacts (fewer than 2 atoms), the model is not
    making any "extra" predictions, thus precision is 1.0.
    """
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    structure = struc.array([h1])
    # Need at least one restraint to bypass the 'if not restraints' check
    restraints = [{"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}]
    scores = calculate_rpf_score(structure, restraints)

    assert scores["precision"] == 1.0
    assert scores["recall"] == 0.0  # Atom 2 is missing


def test_rpf_accepts_seq_keys() -> None:
    """calculate_rpf_score must accept the seq_1/atom_name_1/seq_2/atom_name_2/upper_limit
    key format produced by calculate_synthetic_noes().

    BACKGROUND:
    calculate_synthetic_noes() in synth-nmr returns dicts with seq_1/seq_2 keys.
    A previous bug used falsy `or` chaining which silently accepted the user-friendly
    res_i/res_j format while breaking the engine's own output format.
    This test pins the correct behaviour as a regression guard.
    """
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    # Engine-style keys (as returned by calculate_synthetic_noes)
    restraint = {
        "seq_1": 1,
        "atom_name_1": "H",
        "seq_2": 2,
        "atom_name_2": "H",
        "upper_limit": 5.0,
    }
    scores = calculate_rpf_score(structure, [restraint], distance_threshold=5.0)

    # Both atoms are within 3 Å; the restraint (upper_limit=5 Å) is satisfied.
    assert scores["recall"] == 1.0, f"recall={scores['recall']} — seq_1/seq_2 keys not parsed"
    assert scores["precision"] == 1.0


def test_rpf_res_id_zero_not_dropped() -> None:
    """Residue id=0 must not be silently discarded by falsy `or` chaining.

    BACKGROUND:
    Some PDB files number the N-terminal residue as 0.  The old code used
    `res.get("seq_1") or res.get("res_i")`, which evaluates 0 as falsy and
    falls through to the next key, incorrectly dropping the residue.
    """
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=0, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    restraint = {
        "seq_1": 0,  # ← falsy value that was previously dropped
        "atom_name_1": "H",
        "seq_2": 1,
        "atom_name_2": "H",
        "upper_limit": 5.0,
    }
    scores = calculate_rpf_score(structure, [restraint], distance_threshold=5.0)
    assert scores["recall"] == 1.0, "res_id=0 was silently dropped (falsy `or` bug)"
