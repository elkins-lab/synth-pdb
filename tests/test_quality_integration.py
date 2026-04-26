from unittest.mock import MagicMock, patch

import numpy as np

from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

# Mock a simple dimer PDB: Two ALA residues, one on Chain A, one on Chain B.
DIMER_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   1.200   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       1.400   2.200   0.000  1.00  0.00           O\n"
    "TER       5      ALA A   1\n"
    "ATOM      6  N   ALA B   2      10.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  ALA B   2      11.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   ALA B   2      12.000   1.200   0.000  1.00  0.00           C\n"
    "ATOM      9  O   ALA B   2      11.400   2.200   0.000  1.00  0.00           O\n"
    "TER      10      ALA B   2\n"
)


class TestQualityIntegration:
    """Integrated Quality Scoring (Scientific Defense Scorecard) Tests."""

    def test_quality_report_basic(self) -> None:
        """Verify report returns basic metrics without ML/NMR."""
        # Use real generator to avoid OpenMM issues
        pdb_content = generate_pdb_content(length=5, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)
        report = validator.get_quality_report()

        assert "potential_energy_kj_mol" in report
        assert "is_overall_scientifically_defensible" in report
        assert "ml_score" not in report
        assert "nmr_stats" not in report
        assert "interface_metrics" not in report

    @patch("synth_pdb.quality.classifier.ProteinQualityClassifier")
    def test_quality_report_with_ml(self, mock_clf_class: MagicMock) -> None:
        """Verify report includes ML scores when requested."""
        mock_clf = mock_clf_class.return_value
        mock_clf.model = MagicMock()
        mock_clf.predict.return_value = (True, 0.95, {})

        pdb_content = generate_pdb_content(length=5, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)
        report = validator.get_quality_report(include_ml=True)

        assert report["ml_is_plausible"] is True
        assert report["ml_score"] == 0.95
        assert "nmr_stats" not in report

    def test_quality_report_with_nmr(self) -> None:
        """Verify report includes NMR satisfaction metrics."""
        # Use real generator to ensure geometry passes, then add NMR check
        pdb_content = generate_pdb_content(length=10, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)
        atoms = validator.get_atoms()
        # Find two atoms to restrain
        a1 = atoms[0]
        a2 = atoms[10]
        dist = float(np.linalg.norm(a1["coords"] - a2["coords"]))

        # Satisfied restraint
        restraints = [
            {
                "index_1": a1["residue_number"],
                "atom_name_1": a1["atom_name"],
                "index_2": a2["residue_number"],
                "atom_name_2": a2["atom_name"],
                "upper_limit": dist + 1.0,
            }
        ]

        report = validator.get_quality_report(nmr_restraints=restraints)

        assert "nmr_stats" in report
        assert report["nmr_stats"]["noe_satisfaction_pct"] == 100.0
        assert report["nmr_stats"]["noe_violation_count"] == 0

    def test_quality_report_multichain(self) -> None:
        """Verify report automatically includes interface metrics for multichain."""
        # Patch energy because DIMER_PDB is a mock and will fail simulation
        with patch.object(PDBValidator, "calculate_potential_energy", return_value=0.0):
            validator = PDBValidator(pdb_content=DIMER_PDB)
            report = validator.get_quality_report()

            assert "interface_metrics" in report
            assert "buried_surface_area" in report["interface_metrics"]
            assert report["interface_metrics"]["is_interface_physically_plausible"] is True

    @patch("synth_pdb.quality.classifier.ProteinQualityClassifier")
    def test_scientific_defensability_failure_ml(self, mock_clf_class: MagicMock) -> None:
        """Verify that a structure fails overall defense if ML judge fails."""
        mock_clf = mock_clf_class.return_value
        mock_clf.model = MagicMock()
        # Even if geometry is perfect, if ML says no, it's not defensible
        mock_clf.predict.return_value = (False, 0.10, {})

        pdb_content = generate_pdb_content(length=10, minimize_energy=True, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)
        report = validator.get_quality_report(include_ml=True)

        assert report["is_overall_scientifically_defensible"] is False

    def test_scientific_defensability_failure_nmr(self) -> None:
        """Verify that a structure fails overall defense if NMR judge fails."""
        pdb_content = generate_pdb_content(length=10, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)

        # Violated restraint (force a 0.0A limit for distant atoms)
        restraints = [
            {
                "index_1": 1,
                "atom_name_1": "N",
                "index_2": 5,
                "atom_name_2": "CA",
                "upper_limit": 0.5,
            }
        ]

        report = validator.get_quality_report(nmr_restraints=restraints)

        assert report["nmr_stats"]["noe_satisfaction_pct"] < 100.0
        assert report["is_overall_scientifically_defensible"] is False

    def test_scientific_defensability_failure_interface(self) -> None:
        """Verify that a structure fails overall defense if interface clashes."""
        clashing_pdb = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "TER       2      ALA A   1\n"
            "ATOM      3  CA  ALA B   1       1.000   0.000   0.000  1.00  0.00           C\n"
            "TER       4      ALA B   1\n"
        )
        validator = PDBValidator(pdb_content=clashing_pdb)

        # Patch energy and z-scores to ensure it passes individual checks but fails on interface
        with (
            patch.object(PDBValidator, "calculate_potential_energy", return_value=0.0),
            patch.object(
                PDBValidator, "get_geometric_z_scores", return_value={"mean_bond_zscore": 0.0}
            ),
        ):
            report = validator.get_quality_report()

            assert report["interface_metrics"]["is_interface_physically_plausible"] is False
            assert report["is_overall_scientifically_defensible"] is False

    def test_full_integrated_scorecard_success(self) -> None:
        """Test a perfect scenario where all 3 judges (Physics, NMR, Interface) agree."""
        validator = PDBValidator(pdb_content=DIMER_PDB)

        # Satisfied NMR restraint (N to CA in same residue is ~1.46)
        restraints = [
            {
                "index_1": 1,
                "atom_name_1": "N",
                "index_2": 1,
                "atom_name_2": "CA",
                "upper_limit": 2.0,
            }
        ]

        # Patch all physics judges to return perfect scores for the mock PDB
        with (
            patch.object(PDBValidator, "calculate_potential_energy", return_value=0.0),
            patch.object(
                PDBValidator, "get_geometric_z_scores", return_value={"mean_bond_zscore": 0.0}
            ),
            patch.object(
                PDBValidator, "get_ramachandran_statistics", return_value={"outlier_pct": 0.0}
            ),
            patch.object(
                PDBValidator,
                "get_rotamer_quality_report",
                return_value={"favored_rotamers_pct": 100.0},
            ),
            patch.object(
                PDBValidator, "calculate_residue_sasa", return_value={"burial_ratio": 1.0}
            ),
            patch("synth_pdb.quality.classifier.ProteinQualityClassifier") as mock_clf_class,
        ):
            mock_clf = mock_clf_class.return_value
            mock_clf.model = MagicMock()
            mock_clf.predict.return_value = (True, 0.99, {})

            report = validator.get_quality_report(include_ml=True, nmr_restraints=restraints)

            assert report["is_overall_scientifically_defensible"] is True
            assert "nmr_stats" in report
            assert "interface_metrics" in report
            assert "ml_score" in report

    def test_missing_ml_model_graceful_handling(self) -> None:
        """Verify that report handles cases where the ML model/joblib is missing."""
        # Use real structure for physics but mock ML failure
        pdb_content = generate_pdb_content(length=10, seed=42)
        with (
            patch("synth_pdb.quality.classifier.ProteinQualityClassifier") as mock_clf_class,
            patch.object(PDBValidator, "calculate_potential_energy", return_value=0.0),
            patch.object(
                PDBValidator, "calculate_residue_sasa", return_value={"burial_ratio": 1.0}
            ),
        ):
            mock_clf = mock_clf_class.return_value
            mock_clf.model = None  # Simulates load failure

            validator = PDBValidator(pdb_content=pdb_content)
            report = validator.get_quality_report(include_ml=True)

            assert "ml_score" not in report
            assert report["is_overall_scientifically_defensible"] is True

    def test_empty_nmr_restraints_handling(self) -> None:
        """Verify that an empty list of restraints doesn't crash or trigger NMR stats."""
        pdb_content = generate_pdb_content(length=5, seed=42)
        validator = PDBValidator(pdb_content=pdb_content)
        with patch.object(PDBValidator, "calculate_potential_energy", return_value=0.0):
            report = validator.get_quality_report(nmr_restraints=[])

            assert "nmr_stats" not in report
            assert report["is_overall_scientifically_defensible"] is True
