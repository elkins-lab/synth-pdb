import warnings
import pytest
from synth_pdb.generator import generate_pdb_content


def test_oxt_residue_numbering_regression():
    """
    Regression test for the OXT fix (d15aae9).
    Ensures that cyclic peptides with energy minimization do not trigger
    'two consecutive residues with same number' warnings from OpenMM.
    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Exact reproduction parameters provided by the user
        generate_pdb_content(length=6, cyclic=True, minimize_energy=True, seed=2)

        # Check for the specific warning text that indicates a regression.
        matching_warnings = [
            str(warning.message)
            for warning in w
            if "two consecutive residues with same number" in str(warning.message)
        ]

        # If we failed, print all warnings for debugging
        if len(matching_warnings) > 0:
            print("\nAll caught warnings:")
            for i, warning in enumerate(w):
                print(f"{i + 1}: {warning.category.__name__}: {warning.message}")

        assert len(matching_warnings) == 0, (
            f"Regression detected: Found 'two consecutive residues with same number' "
            f"warnings: {matching_warnings}"
        )
