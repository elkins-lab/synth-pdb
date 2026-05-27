"""
Tests for PDBValidator.calculate_residue_sasa() and burial_ratio.

Coverage targets:
- Return schema (all expected keys present)
- burial_ratio formula: mean_polar / (mean_hydro + 1e-6)
- Correct classification of hydrophobic vs polar residues
- Edge cases: all-hydrophobic sequence, all-polar sequence, single residue
- Invalid PDB input returns safe fallback
- is_biophysically_plausible threshold (>= 0.8) in get_quality_report
- burial_ratio appears in get_quality_report output
- Directionality: polar-only >> hydrophobic-only burial ratio
"""

import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator(sequence_str: str) -> PDBValidator:
    pdb = generate_pdb_content(sequence_str=sequence_str)
    return PDBValidator(pdb_content=pdb)


# ---------------------------------------------------------------------------
# Schema / contract tests
# ---------------------------------------------------------------------------


class TestCalculateResidueSasaSchema:
    """Verify the return dict always has the expected keys and sane types."""

    def test_returns_all_required_keys(self) -> None:
        v = _make_validator("ALALALA")
        result = v.calculate_residue_sasa()
        assert "SASA" in result
        assert "mean_hydrophobic_sasa" in result
        assert "mean_polar_sasa" in result
        assert "burial_ratio" in result

    def test_sasa_dict_maps_residue_ids_to_floats(self) -> None:
        v = _make_validator("GGGGG")
        result = v.calculate_residue_sasa()
        sasa = result["SASA"]
        assert isinstance(sasa, dict)
        assert len(sasa) == 5
        for res_id, val in sasa.items():
            assert isinstance(res_id, int), f"res_id {res_id} is not int"
            assert isinstance(val, float), f"SASA value {val} is not float"
            assert val >= 0.0, f"SASA cannot be negative, got {val}"

    def test_summary_stats_are_floats(self) -> None:
        v = _make_validator("ALALALA")
        result = v.calculate_residue_sasa()
        assert isinstance(result["mean_hydrophobic_sasa"], float)
        assert isinstance(result["mean_polar_sasa"], float)
        assert isinstance(result["burial_ratio"], float)

    def test_burial_ratio_is_non_negative(self) -> None:
        v = _make_validator("ALALALA")
        result = v.calculate_residue_sasa()
        assert result["burial_ratio"] >= 0.0


# ---------------------------------------------------------------------------
# burial_ratio formula tests
# ---------------------------------------------------------------------------


class TestBurialRatioFormula:
    """
    burial_ratio = mean_polar / (mean_hydro + 1e-6)

    Key invariants:
      - All-polar sequence  -> mean_hydro ~ 0  -> ratio >> 1  (high)
      - All-hydrophobic     -> mean_polar = 1.0 (div-by-zero guard) -> ratio low
      - Mixed sequence      -> ratio somewhere in between
    """

    # Hydrophobic residues per implementation: VAL ILE LEU PHE TRP MET TYR
    # Polar residues: everything else (GLY ALA SER ASP LYS ARG GLN ASN etc.)

    def test_all_polar_gives_high_burial_ratio(self) -> None:
        """A pure polar chain has mean_hydro~0, so burial_ratio should be large."""
        # GLY and ALA are both classified as polar (not in hydrophobic_res list)
        v = _make_validator("GGGGGGGGG")
        result = v.calculate_residue_sasa()
        # With mean_hydro ~ 0, ratio = mean_polar / 1e-6 which is very large
        # In the else-branch mean_polar defaults to 1.0 when polar_vals is empty,
        # but here GLY IS polar so polar_vals will be populated.
        assert result["burial_ratio"] > 1.0, (
            f"All-polar sequence should give burial_ratio > 1.0, got {result['burial_ratio']}"
        )

    def test_all_hydrophobic_gives_lower_burial_ratio_than_all_polar(self) -> None:
        """A pure hydrophobic chain should have a lower burial ratio than a polar chain."""
        v_polar = _make_validator("GGGGGGGGG")  # all GLY  -> polar
        v_hydro = _make_validator("VVVVVVVVV")  # all VAL  -> hydrophobic

        ratio_polar = v_polar.calculate_residue_sasa()["burial_ratio"]
        ratio_hydro = v_hydro.calculate_residue_sasa()["burial_ratio"]

        assert ratio_polar > ratio_hydro, (
            f"Polar burial ratio ({ratio_polar:.3f}) should exceed "
            f"hydrophobic burial ratio ({ratio_hydro:.3f})"
        )

    def test_mean_hydrophobic_zero_when_no_hydrophobic_residues(self) -> None:
        """If there are no hydrophobic residues, mean_hydrophobic_sasa must be 0.0."""
        v = _make_validator("GGGGG")
        result = v.calculate_residue_sasa()
        assert result["mean_hydrophobic_sasa"] == 0.0

    def test_mean_hydrophobic_nonzero_when_hydrophobic_present(self) -> None:
        """With VAL/LEU/ILE in the sequence, mean_hydrophobic_sasa must be > 0."""
        v = _make_validator("VVVVV")
        result = v.calculate_residue_sasa()
        assert result["mean_hydrophobic_sasa"] > 0.0

    def test_burial_ratio_denominator_guard(self) -> None:
        """
        When all residues are polar (mean_hydro=0), the 1e-6 guard prevents
        ZeroDivisionError and produces a finite float.
        """
        v = _make_validator("GGGG")
        result = v.calculate_residue_sasa()
        assert result["burial_ratio"] != float("inf")
        assert result["burial_ratio"] == result["burial_ratio"]  # not NaN


# ---------------------------------------------------------------------------
# Residue classification tests
# ---------------------------------------------------------------------------


class TestHydrophobicClassification:
    """
    The seven hydrophobic residues in the implementation:
    VAL, ILE, LEU, PHE, TRP, MET, TYR
    Everything else should land in polar_vals.
    """

    @pytest.mark.parametrize(
        "one_letter,three_letter",
        [
            ("V", "VAL"),
            ("I", "ILE"),
            ("L", "LEU"),
            ("F", "PHE"),
            ("W", "TRP"),
            ("M", "MET"),
            ("Y", "TYR"),
        ],
    )
    def test_each_hydrophobic_residue_raises_mean_hydrophobic_sasa(
        self, one_letter: str, three_letter: str
    ) -> None:
        """A single hydrophobic residue surrounded by GLY should register hydro SASA."""
        # Flank with GLY so we have enough residues for a valid structure
        seq = f"GGG{one_letter}GGG"
        v = _make_validator(seq)
        result = v.calculate_residue_sasa()
        assert result["mean_hydrophobic_sasa"] > 0.0, (
            f"{three_letter} was not classified as hydrophobic (mean_hydrophobic_sasa=0)"
        )

    def test_gly_ala_ser_classified_as_polar(self) -> None:
        """GLY, ALA, SER should NOT appear in hydro_vals - only in polar."""
        v = _make_validator("GASGAS")
        result = v.calculate_residue_sasa()
        # Because no hydrophobic residues: mean_hydrophobic_sasa must be 0
        assert result["mean_hydrophobic_sasa"] == 0.0


# ---------------------------------------------------------------------------
# Edge-case / robustness tests
# ---------------------------------------------------------------------------


class TestSasaEdgeCases:
    """Boundary and fault-tolerance scenarios."""

    def test_single_residue(self) -> None:
        """A single-residue structure should not crash and return sane values."""
        v = _make_validator("ALA")
        result = v.calculate_residue_sasa()
        assert "SASA" in result
        assert len(result["SASA"]) == 1
        assert result["burial_ratio"] >= 0.0

    def test_invalid_pdb_content_returns_safe_fallback(self) -> None:
        """Completely invalid PDB content should return the fallback dict, not raise."""
        v = PDBValidator(pdb_content="ATOM  garbage line\n")
        result = v.calculate_residue_sasa()
        # Should return the safe fallback dict
        assert "SASA" in result
        assert "mean_hydrophobic_sasa" in result

    def test_empty_pdb_returns_safe_fallback(self) -> None:
        """An empty-but-valid PDB string should not raise."""
        v = PDBValidator(pdb_content="END\n")
        result = v.calculate_residue_sasa()
        assert isinstance(result, dict)
        assert "SASA" in result

    def test_sasa_values_sum_positive_for_real_structure(self) -> None:
        """Total SASA across all residues must be > 0 for a real structure."""
        v = _make_validator("ALALALA")
        result = v.calculate_residue_sasa()
        total = sum(result["SASA"].values())
        assert total > 0.0, f"Total SASA should be positive, got {total}"

    def test_longer_sequence_has_correct_residue_count(self) -> None:
        """SASA dict should have one entry per residue."""
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
        v = _make_validator(seq)
        result = v.calculate_residue_sasa()
        assert len(result["SASA"]) == 20


# ---------------------------------------------------------------------------
# Integration with get_quality_report
# ---------------------------------------------------------------------------


class TestBurialRatioInQualityReport:
    """Verify burial_ratio is surfaced correctly in get_quality_report."""

    def test_quality_report_contains_burial_ratio(self) -> None:
        v = _make_validator("ALALALA")
        report = v.get_quality_report()
        assert "hydrophobic_burial_ratio" in report
        assert isinstance(report["hydrophobic_burial_ratio"], float)

    def test_quality_report_contains_is_biophysically_plausible(self) -> None:
        v = _make_validator("ALALALA")
        report = v.get_quality_report()
        assert "is_biophysically_plausible" in report
        assert isinstance(report["is_biophysically_plausible"], bool)

    def test_is_biophysically_plausible_reflects_burial_ratio_threshold(self) -> None:
        """
        is_biophysically_plausible = burial_ratio >= 0.8
        We verify this is consistent - not that a specific value passes,
        since linear peptides may not reach 0.8.
        """
        v = _make_validator("ALALALA")
        report = v.get_quality_report()
        ratio = report["hydrophobic_burial_ratio"]
        expected_plausible = ratio >= 0.8
        assert report["is_biophysically_plausible"] == expected_plausible, (
            f"burial_ratio={ratio:.3f} but is_biophysically_plausible="
            f"{report['is_biophysically_plausible']} (expected {expected_plausible})"
        )

    def test_polar_sequence_has_higher_burial_ratio_in_report_than_hydrophobic(self) -> None:
        """Polar sequences should score higher on burial_ratio in the report."""
        report_polar = _make_validator("GGGGGGG").get_quality_report()
        report_hydro = _make_validator("VVVVVVV").get_quality_report()

        assert (
            report_polar["hydrophobic_burial_ratio"] > report_hydro["hydrophobic_burial_ratio"]
        ), (
            f"Polar burial ratio ({report_polar['hydrophobic_burial_ratio']:.3f}) "
            f"should exceed hydrophobic ({report_hydro['hydrophobic_burial_ratio']:.3f})"
        )

    def test_burial_ratio_consistent_between_direct_call_and_report(self) -> None:
        """burial_ratio from calculate_residue_sasa() should match what's in the report."""
        v = _make_validator("ALALALA")
        direct = v.calculate_residue_sasa()["burial_ratio"]
        report = v.get_quality_report()["hydrophobic_burial_ratio"]
        assert abs(direct - report) < 1e-9, (
            f"Direct call returned {direct:.6f}, report returned {report:.6f}"
        )
