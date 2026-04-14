import biotite.structure as struc
import pytest

from synth_pdb.nmr import calculate_rpf_score


def test_rpf_perfect_fit():
    """If all restraints are satisfied and no extra short distances exist, RPF should be 1.0."""
    # Create a simple structure: two protons at 3.0 A
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    # Create a restraint for them
    restraints = [
        {"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}
    ]

    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["f_measure"] == 1.0

def test_rpf_unsatisfied_restraint():
    """If a restraint is violated (distance > upper bound), recall should decrease."""
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([6, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2])

    # Restraint says < 5.0, but it's 6.0
    restraints = [
        {"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}
    ]

    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 0.0
    assert scores["f_measure"] == 0.0

def test_rpf_low_precision():
    """If the structure has many short distances not in the restraints, precision should be low."""
    # Three protons in a triangle, all at 3.0 A
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    h2 = struc.Atom([3, 0, 0], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A")
    h3 = struc.Atom([0, 3, 0], atom_name="H", element="H", res_id=3, res_name="ALA", chain_id="A")
    structure = struc.array([h1, h2, h3])

    # Only one restraint (H1-H2)
    restraints = [
        {"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}
    ]

    # Structure has 3 short distances: (1,2), (1,3), (2,3)
    # Only (1,2) is in restraints.
    # Precision = 1/3 = 0.33
    scores = calculate_rpf_score(structure, restraints, distance_threshold=5.0)

    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(0.333, abs=0.01)
    # F = 2 * (1 * 1/3) / (1 + 1/3) = (2/3) / (4/3) = 0.5
    assert scores["f_measure"] == pytest.approx(0.5, abs=0.01)

def test_rpf_empty_restraints():
    """Empty restraints should return zeros."""
    h1 = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A")
    structure = struc.array([h1])
    scores = calculate_rpf_score(structure, [])
    assert scores == {"recall": 0.0, "precision": 0.0, "f_measure": 0.0}
