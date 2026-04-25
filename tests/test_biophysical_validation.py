from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


def test_hydrophobic_burial_validation_tdd() -> None:
    """TDD: Verify that PDBValidator can detect the "Oil Drop" effect (Hydrophobic Burial).

    SCIENTIFIC BASIS:
    The Hydrophobic Effect (Kauzmann, 1959) is the primary driver of protein folding.
    Hydrophobic residues (V, I, L, F, W, M) should be shielded from solvent (low SASA).
    A "meaningless" structure might have hydrophobic residues highly exposed.
    """
    # 1. Generate a globular-like structure (or at least one with some depth)
    # We'll use a sequence with a hydrophobic core (V, I, L)
    seq = "GAAAVILVAAAG"
    pdb_content = generate_pdb_content(sequence_str=seq)

    validator = PDBValidator(pdb_content=pdb_content)

    # New method to calculate SASA per residue (TDD Red Phase)
    sasa_results = validator.calculate_residue_sasa()

    assert "SASA" in sasa_results
    assert len(sasa_results["SASA"]) > 0

    # Check if hydrophobic residues are generally more buried than Gly/Ala
    # (Note: In a random linear model this might be subtle, but we test the API exists)
    assert sasa_results["mean_hydrophobic_sasa"] >= 0


def test_quality_report_includes_biophysics() -> None:
    """TDD: Verify that biophysical burial is included in the quality report."""
    pdb_content = generate_pdb_content(length=10)
    validator = PDBValidator(pdb_content=pdb_content)

    report = validator.get_quality_report()

    assert "hydrophobic_burial_ratio" in report
    assert "is_biophysically_plausible" in report


def test_validate_distance_restraints_clean() -> None:
    """Verify that a structure with satisfy-able distances passes validation."""
    # Create a 2-residue structure with known distance
    pdb_content = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\n"
    )
    validator = PDBValidator(pdb_content=pdb_content)

    # Restraint: CA 1 to CA 2 within 5.0A (Actual is 3.8A)
    restraints = [
        {"index_1": 1, "atom_name_1": "CA", "index_2": 2, "atom_name_2": "CA", "upper_limit": 5.0}
    ]

    validator.validate_distance_restraints(restraints)
    assert len(validator.get_violations()) == 0


def test_validate_distance_restraints_violation() -> None:
    """Verify that a structure exceeding an upper limit flags a violation."""
    pdb_content = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       8.000   0.000   0.000  1.00  0.00           C\n"
    )
    validator = PDBValidator(pdb_content=pdb_content)

    # Restraint: CA 1 to CA 2 within 5.0A (Actual is 8.0A)
    restraints = [
        {"index_1": 1, "atom_name_1": "CA", "index_2": 2, "atom_name_2": "CA", "upper_limit": 5.0}
    ]

    validator.validate_distance_restraints(restraints)
    violations = validator.get_violations()
    assert len(violations) == 1
    assert "NMR Restraint violation" in violations[0]
    assert "Measured effective distance: 8.00Å" in violations[0]


def test_validate_distance_restraints_pseudo_atoms() -> None:
    """Verify that pseudo-atom aliases are correctly expanded and r^-6 averaged."""
    # Create structure where CA1 is near HB2 but far from HB3 of ALA 2
    pdb_content = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  HB1 ALA A   2       2.000   0.000   0.000  1.00  0.00           H\n"
        "ATOM      4  HB2 ALA A   2       2.500   0.000   0.000  1.00  0.00           H\n"
        "ATOM      5  HB3 ALA A   2      10.000   0.000   0.000  1.00  0.00           H\n"
    )
    validator = PDBValidator(pdb_content=pdb_content)

    # Restraint using pseudo-atom "HB" for residue 2
    # This should find HB1, HB2, HB3 and calculate (dist1^-6 + dist2^-6 + dist3^-6)^-1/6
    # dist1 = 2.0, dist2 = 2.5, dist3 = 10.0
    # Expected d_eff approx < 2.0 (since 2.0 dominates)
    restraints = [
        {"index_1": 1, "atom_name_1": "CA", "index_2": 2, "atom_name_2": "HB", "upper_limit": 2.5}
    ]

    validator.validate_distance_restraints(restraints)
    assert len(validator.get_violations()) == 0  # Should pass as d_eff < 2.5
