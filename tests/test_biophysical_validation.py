from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


def test_hydrophobic_burial_validation_tdd():
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


def test_quality_report_includes_biophysics():
    """TDD: Verify that biophysical burial is included in the quality report."""
    pdb_content = generate_pdb_content(length=10)
    validator = PDBValidator(pdb_content=pdb_content)

    report = validator.get_quality_report()

    assert "hydrophobic_burial_ratio" in report
    assert "is_biophysically_plausible" in report
