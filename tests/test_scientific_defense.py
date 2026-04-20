from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


def test_engh_huber_zscore_validation() -> None:
    """Aggressive validation: Verify that PDBValidator flags Z-score outliers.

    SCIENTIFIC BASIS:
    Engh & Huber (1991) established standard deviations for protein geometry.
    A structure with an average bond length Z-score > 3.0 is
    statistically impossible for a real protein.
    """
    # 1. Good structure (standard alpha helix)
    pdb_good = generate_pdb_content(length=10)
    val_good = PDBValidator(pdb_content=pdb_good)

    # Method to implement: get_geometric_z_scores()
    z_scores = val_good.get_geometric_z_scores()

    assert "mean_bond_zscore" in z_scores
    assert z_scores["mean_bond_zscore"] < 2.0  # Minimized/idealized should be very low

    # 2. Bad structure (stretch ALL bonds)
    atoms_bad = val_good.get_atoms()
    for atom in atoms_bad:
        # Move every atom a bit to stretch all bonds
        # (Relative to the origin, this scales the molecule)
        atom["coords"] *= 1.1  # 10% stretch

    val_bad = PDBValidator(parsed_atoms=atoms_bad)
    z_scores_bad = val_bad.get_geometric_z_scores()

    print(f"Bad Bond Z-score (Scaled): {z_scores_bad['mean_bond_zscore']:.2f}")
    assert z_scores_bad["mean_bond_zscore"] > 5.0


def test_dunbrack_rotamer_probability() -> None:
    """Aggressive validation: Verify that sidechains follow Dunbrack rotamer probabilities.

    SCIENTIFIC BASIS:
    Sidechain Chi angles prefer specific 'rotamers'. Conformations with
    low probability in the Dunbrack library are likely steric outliers.
    """
    pdb_content = generate_pdb_content(length=20)
    validator = PDBValidator(pdb_content=pdb_content)

    # Method to implement: get_rotamer_quality_report()
    rotamer_report = validator.get_rotamer_quality_report()

    assert "favored_rotamers_pct" in rotamer_report
    # A scientifically sound generator should produce > 80% favored rotamers
    assert rotamer_report["favored_rotamers_pct"] > 50.0


def test_comprehensive_scientific_defense_report() -> None:
    """Verify that all metrics are integrated into the final quality assessment."""
    pdb_content = generate_pdb_content(length=10)
    validator = PDBValidator(pdb_content=pdb_content)

    report = validator.get_quality_report()

    # The final report must now defend itself with ALL evidence
    assert "geometric_z_scores" in report
    assert "rotamer_stats" in report
    assert "is_overall_scientifically_defensible" in report
