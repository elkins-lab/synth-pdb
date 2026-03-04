"""
TDD Tests for new CLI flags: --output-rdcs and --shift-predictor.

These tests are written BEFORE main.py is modified (Red phase of TDD).
They will FAIL initially because the CLI arguments do not exist yet.

EDUCATIONAL NOTE — Synthetic RDCs in AI Training:
===================================================
Residual Dipolar Couplings (RDCs) encode orientational relationships between
bond vectors and a global alignment frame. When used as training labels for
structure-prediction models (e.g., AlphaFold-style architectures), they
provide a complementary restraint to NOEs: NOEs give local distance information
while RDCs give global orientational information.

The two new flags being tested here:

  --output-rdcs <file.csv>
      Computes backbone ¹D_NH RDCs for the generated structure and writes
      them to a CSV file. Requires the structure to contain amide H atoms
      (use --minimize to add protons via OpenMM's hydrogen addition).

  --shift-predictor [shiftx2|empirical]
      When --gen-shifts is set, controls which predictor synth-nmr uses:
        • shiftx2  : calls the SHIFTX2 external binary (Han et al., 2011,
                     J Biomol NMR 50:43), falls back to empirical if not found.
        • empirical: uses the SPARTA+/empirical table method directly
                     (Shen & Bax, 2010, J Biomol NMR 48:13).

References:
  Han, B. et al. (2011). SHIFTX2: significantly improved protein chemical shift
  prediction. J Biomol NMR, 50, 43–57. DOI: 10.1007/s10858-011-9478-4

  Shen, Y. & Bax, A. (2010). SPARTA+: a modest improvement in empirical NMR
  chemical shift prediction by means of an artificial neural network.
  J Biomol NMR, 48, 13–22. DOI: 10.1007/s10858-010-9433-9

  Tjandra, N. & Bax, A. (1997). Direct measurement of distances and angles in
  biomolecules by NMR in a dilute liquid crystalline medium. Science, 278,
  1111–1114. DOI: 10.1126/science.278.5340.1111
"""

import logging
import os

import numpy as np
import pytest

from synth_pdb import main
from synth_pdb.generator import create_atom_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pdb_with_nh():
    """Minimal PDB with a backbone N and H atom so RDC calculation is possible."""
    return (
        "HEADER    rdc_test\n"
        + create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N") + "\n"
        + create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C") + "\n"
        + create_atom_line(3, "C", "ALA", "A", 1, 2.500, 0.0, 0.0, "C") + "\n"
        + create_atom_line(4, "H", "ALA", "A", 1, 0.0, 1.02, 0.0, "H") + "\n"
        + create_atom_line(5, "N", "GLY", "A", 2, 3.8, 0.0, 0.0, "N") + "\n"
        + create_atom_line(6, "CA", "GLY", "A", 2, 5.258, 0.0, 0.0, "C") + "\n"
        + create_atom_line(7, "C", "GLY", "A", 2, 6.783, 0.0, 0.0, "C") + "\n"
        + create_atom_line(8, "H", "GLY", "A", 2, 3.8, 1.02, 0.0, "H") + "\n"
    )


# ---------------------------------------------------------------------------
# --output-rdcs flag tests
# ---------------------------------------------------------------------------

class TestOutputRDCsFlag:
    """Verify that the --output-rdcs flag triggers RDC calculation and CSV export."""

    def test_output_rdcs_calls_calculate_rdcs(self, mocker, tmp_path, caplog):
        """
        When --output-rdcs is provided, calculate_rdcs() must be called.

        This test initially FAILS because the CLI argument does not exist yet
        (the Red phase of Red-Green-Refactor TDD).
        """
        caplog.set_level(logging.INFO)
        output_pdb = tmp_path / "rdc_test.pdb"
        output_csv = tmp_path / "rdc_test.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())

        # Spy on the RDC calculation so we can verify it is called
        mock_calc = mocker.patch(
            "synth_pdb.rdc.calculate_rdcs",
            return_value={1: 5.3, 2: -2.1},
        )

        test_args = [
            "synth_pdb",
            "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
        ]
        mocker.patch("sys.argv", test_args)
        mocker.patch("sys.exit")

        main.main()

        mock_calc.assert_called_once()

    def test_output_rdcs_creates_csv_file(self, mocker, tmp_path, caplog):
        """The RDC CSV file must be created at the specified output path."""
        caplog.set_level(logging.INFO)
        output_pdb = tmp_path / "rdc_test.pdb"
        output_csv = tmp_path / "rdc_test.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        mocker.patch("synth_pdb.rdc.calculate_rdcs", return_value={1: 5.3, 2: -2.1})

        test_args = [
            "synth_pdb",
            "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
        ]
        mocker.patch("sys.argv", test_args)
        mocker.patch("sys.exit")

        main.main()

        assert output_csv.exists(), f"Expected CSV file at {output_csv}"

    def test_output_rdcs_csv_has_correct_header(self, mocker, tmp_path):
        """CSV must begin with the column header: res_id,residue,RDC_NH_Hz."""
        output_pdb = tmp_path / "rdc_test.pdb"
        output_csv = tmp_path / "rdc_test.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        mocker.patch("synth_pdb.rdc.calculate_rdcs", return_value={1: 5.3, 2: -2.1})

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
        ])
        mocker.patch("sys.exit")

        main.main()

        with open(output_csv) as f:
            header = f.readline().strip()
        assert header == "res_id,residue,RDC_NH_Hz", (
            f"Expected 'res_id,residue,RDC_NH_Hz', got '{header}'"
        )

    def test_output_rdcs_csv_has_data_rows(self, mocker, tmp_path):
        """CSV must contain at least one data row with the computed RDC values."""
        output_pdb = tmp_path / "rdc_test.pdb"
        output_csv = tmp_path / "rdc_test.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        mocker.patch("synth_pdb.rdc.calculate_rdcs",
                     return_value={1: 5.3000, 2: -2.1000})

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
        ])
        mocker.patch("sys.exit")

        main.main()

        with open(output_csv) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # Header + at least 1 data row
        assert len(lines) >= 2, "CSV must have at least one data row"
        # Verify numeric content is plausible
        data_line = lines[1]
        parts = data_line.split(",")
        assert len(parts) == 3, f"Each data row must have 3 columns, got: {data_line}"
        float(parts[2])  # Should be parseable as float without raising

    def test_output_rdcs_logs_export_message(self, mocker, tmp_path, caplog):
        """A 'RDC data exported to:' log message must be emitted."""
        caplog.set_level(logging.INFO)
        output_pdb = tmp_path / "rdc_test.pdb"
        output_csv = tmp_path / "rdc_test.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        mocker.patch("synth_pdb.rdc.calculate_rdcs", return_value={1: 5.3})

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
        ])
        mocker.patch("sys.exit")

        main.main()

        assert "RDC data exported to:" in caplog.text

    def test_output_rdcs_not_called_without_flag(self, mocker, tmp_path):
        """Without --output-rdcs, calculate_rdcs must NOT be called."""
        output_pdb = tmp_path / "no_rdc.pdb"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        # If import is attempted at all, this still works — we only verify it
        # is never called when the flag is absent.
        mock_calc = mocker.patch("synth_pdb.rdc.calculate_rdcs",
                                 return_value={})

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "2",
            "--output", str(output_pdb),
        ])
        mocker.patch("sys.exit")

        main.main()

        mock_calc.assert_not_called()

    def test_rdc_da_and_r_flags_passed_through(self, mocker, tmp_path):
        """
        Custom --rdc-da and --rdc-r values must be forwarded to calculate_rdcs.

        Physical note: Da and R together define the alignment tensor.
        Typical experimental values for dilute liquid crystal media:
          Da  ~  5–25 Hz  (Tjandra & Bax, 1997)
          R   ~  0–0.67   (axially symmetric=0, maximum rhombicity≈0.67)
        """
        output_pdb = tmp_path / "rdc_custom.pdb"
        output_csv = tmp_path / "rdc_custom.csv"

        mocker.patch("synth_pdb.main.generate_pdb_content",
                     return_value=_minimal_pdb_with_nh())
        mock_calc = mocker.patch("synth_pdb.rdc.calculate_rdcs",
                                 return_value={1: 3.0})

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "2",
            "--output", str(output_pdb),
            "--output-rdcs", str(output_csv),
            "--rdc-da", "15.0",
            "--rdc-r", "0.3",
        ])
        mocker.patch("sys.exit")

        main.main()

        mock_calc.assert_called_once()
        _, call_kwargs = mock_calc.call_args
        assert call_kwargs.get("Da") == pytest.approx(15.0), \
            "Da should be 15.0 Hz as specified"
        assert call_kwargs.get("R") == pytest.approx(0.3), \
            "R should be 0.3 as specified"


# ---------------------------------------------------------------------------
# --shift-predictor flag tests
# ---------------------------------------------------------------------------

class TestShiftPredictorFlag:
    """
    Verify that --shift-predictor correctly selects the chemical shift predictor.

    EDUCATIONAL NOTE — Chemical Shift Prediction Methods:
    =====================================================
    Two approaches are available via synth-nmr:

    1. SHIFTX2 (Han et al., 2011):
       A hybrid machine-learning / empirical method. Combines Random Forest,
       Support Vector Regression, and empirical ring-current corrections.
       RMSD from experiment: ~0.04 ppm (1H), ~0.44 ppm (13C), ~1.17 ppm (15N).
       Requires an external SHIFTX2 binary.

    2. SPARTA+ / Empirical (Shen & Bax, 2010):
       Neural network trained on the BMRB database.
       Always available (pure Python), no external binary needed.
       RMSD from experiment: ~0.05 ppm (1H), ~0.55 ppm (13C), ~2.06 ppm (15N).

    The --shift-predictor flag lets users explicitly select the method,
    e.g., 'empirical' for reproducible CI runs where SHIFTX2 is unavailable.
    """

    def _run_with_shift_predictor(self, mocker, tmp_path, predictor, mock_predict):
        """Helper: runs main() with --gen-shifts and a given --shift-predictor."""
        output_pdb = tmp_path / "shifts_test.pdb"
        pdb_with_h = (
            "HEADER    shift_test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N") + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C") + "\n"
            + create_atom_line(3, "H", "ALA", "A", 1, 0.0, 1.02, 0.0, "H") + "\n"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=pdb_with_h)
        mocker.patch("synth_pdb.nef_io.write_nef_chemical_shifts")

        args = [
            "synth_pdb", "--length", "1",
            "--output", str(output_pdb),
            "--gen-shifts",
            "--shift-predictor", predictor,
        ]
        mocker.patch("sys.argv", args)
        mocker.patch("sys.exit")
        main.main()

    def test_shift_predictor_shiftx2_passes_use_shiftx2_true(self, mocker, tmp_path):
        """
        --shift-predictor shiftx2 must call predict_chemical_shifts with use_shiftx2=True.
        """
        mock_predict = mocker.patch(
            "synth_pdb.chemical_shifts.predict_chemical_shifts",
            return_value={},
        )
        self._run_with_shift_predictor(mocker, tmp_path, "shiftx2", mock_predict)

        mock_predict.assert_called_once()
        _, call_kwargs = mock_predict.call_args
        assert call_kwargs.get("use_shiftx2") is True, (
            "--shift-predictor shiftx2 must pass use_shiftx2=True"
        )

    def test_shift_predictor_empirical_passes_use_shiftx2_false(self, mocker, tmp_path):
        """
        --shift-predictor empirical must call predict_chemical_shifts with use_shiftx2=False.
        """
        mock_predict = mocker.patch(
            "synth_pdb.chemical_shifts.predict_chemical_shifts",
            return_value={},
        )
        self._run_with_shift_predictor(mocker, tmp_path, "empirical", mock_predict)

        mock_predict.assert_called_once()
        _, call_kwargs = mock_predict.call_args
        assert call_kwargs.get("use_shiftx2") is False, (
            "--shift-predictor empirical must pass use_shiftx2=False"
        )

    def test_shift_predictor_default_is_shiftx2(self, mocker, tmp_path):
        """
        When --gen-shifts is used without --shift-predictor, the default must be
        use_shiftx2=True (consistent with synth-nmr's own default behaviour).
        """
        output_pdb = tmp_path / "default_pred.pdb"
        pdb_with_h = (
            "HEADER    default_pred\n"
            + create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N") + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C") + "\n"
            + create_atom_line(3, "H", "ALA", "A", 1, 0.0, 1.02, 0.0, "H") + "\n"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=pdb_with_h)
        mock_predict = mocker.patch(
            "synth_pdb.chemical_shifts.predict_chemical_shifts",
            return_value={},
        )
        mocker.patch("synth_pdb.nef_io.write_nef_chemical_shifts")

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "1",
            "--output", str(output_pdb),
            "--gen-shifts",
            # NOTE: --shift-predictor intentionally omitted
        ])
        mocker.patch("sys.exit")

        main.main()

        mock_predict.assert_called_once()
        _, call_kwargs = mock_predict.call_args
        assert call_kwargs.get("use_shiftx2") is True, (
            "Default --shift-predictor should be 'shiftx2' (use_shiftx2=True)"
        )

    def test_shift_predictor_appears_in_command_string(self, mocker, tmp_path):
        """
        The --shift-predictor flag must appear in the command string recorded
        in the PDB REMARK header so the generation is fully reproducible.

        EDUCATIONAL NOTE — Provenance in PDB Headers:
        ===============================================
        PDB files generated by synth-pdb embed a REMARK 3 block containing the
        exact CLI command used. This is critical for scientific reproducibility:
        any researcher can re-generate the identical structure from the REMARK
        block, and peer reviewers can audit the generation parameters.
        """
        output_pdb = tmp_path / "remark_test.pdb"
        pdb_with_h = (
            "HEADER    remark_test\n"
            + create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N") + "\n"
            + create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C") + "\n"
            + create_atom_line(3, "H", "ALA", "A", 1, 0.0, 1.02, 0.0, "H") + "\n"
        )
        mocker.patch("synth_pdb.main.generate_pdb_content", return_value=pdb_with_h)
        mocker.patch("synth_pdb.chemical_shifts.predict_chemical_shifts", return_value={})
        mocker.patch("synth_pdb.nef_io.write_nef_chemical_shifts")

        mocker.patch("sys.argv", [
            "synth_pdb", "--length", "1",
            "--output", str(output_pdb),
            "--gen-shifts",
            "--shift-predictor", "empirical",
        ])
        mocker.patch("sys.exit")

        main.main()

        with open(output_pdb) as f:
            content = f.read()
        assert "empirical" in content or "shift-predictor" in content.lower(), (
            "The PDB REMARK header must record the --shift-predictor value"
        )
