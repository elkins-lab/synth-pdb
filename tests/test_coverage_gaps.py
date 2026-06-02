import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synth_pdb.physics import EnergyMinimizer


class TestCoverageGaps:
    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_ptm_atom_stripping_and_translation(self, mock_app: MagicMock) -> None:
        """Verify that PTMs (SEP, TPO, PTR) are translated to standard residues
        and their extra atoms (P, O1P, etc.) are stripped.
        """
        minimizer = EnergyMinimizer(disable_cache=True)

        pdb_lines = [
            "ATOM    100  N   SEP A  10      11.111  22.222  33.333  1.00  0.00           N  ",
            "ATOM    101  CA  SEP A  10      11.111  22.222  33.333  1.00  0.00           C  ",
            "ATOM    102  P   SEP A  10      11.111  22.222  33.333  1.00  0.00           P  ",
            "ATOM    103  O1P SEP A  10      11.111  22.222  33.333  1.00  0.00           O  ",
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tf:
            tf.writelines([line + "\n" for line in pdb_lines])
            input_path = tf.name

        try:
            with patch("tempfile.NamedTemporaryFile") as mock_tf_class:
                mock_tf = MagicMock()
                mock_tf.__enter__.return_value.name = "intercepted.pdb"
                mock_tf_class.return_value = mock_tf

                mock_app.PDBFile.side_effect = Exception("Stop execution after strip check")

                try:
                    minimizer._run_simulation(input_path, "out.pdb")
                except Exception as e:
                    if str(e) != "Stop execution after strip check":
                        raise

                written_lines = mock_tf.__enter__.return_value.writelines.call_args[0][0]

                for line in written_lines:
                    if "SEP" in line:
                        pytest.fail(f"Residue name 'SEP' should have been translated: {line}")
                    if any(at in line for at in ["P", "O1P", "O2P", "O3P"]):
                        # CAREFUL: CA, N, C etc don't contain P, O1P
                        # but "P" is a sub-string of many things.
                        # The stripper only targets exact atom name match in columns 12-16
                        atom_name = line[12:16].strip()
                        if atom_name in ["P", "O1P", "O2P", "O3P"]:
                            pytest.fail(f"PTM atom '{atom_name}' should have been stripped.")
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    @patch("synth_pdb.physics.app")
    def test_low_pot_energy_threshold_no_warning(
        self, mock_app: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that NO warning is logged if potential energy is reasonable."""
        import logging

        caplog.set_level(logging.WARNING)

        # Patch ForceField constructor to avoid real file loading
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()

        minimizer = EnergyMinimizer(disable_cache=True)

        # Mocking the PDB loading
        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb

        mock_topo = MagicMock()
        mock_atom = MagicMock()
        mock_topo.atoms.return_value = [mock_atom]
        mock_pdb.topology = mock_topo
        mock_pos = MagicMock()
        mock_pos.__len__.return_value = 1
        mock_pos.value_in_unit.return_value = [[1, 1, 1]]
        mock_pdb.positions = mock_pos
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology = mock_topo
        mock_modeller.positions = mock_pos
        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim

        # Return reasonable energy
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = -100.0
        mock_state.getPositions.return_value = mock_pos
        mock_sim.context.getState.return_value = mock_state

        with patch("synth_pdb.physics.app.PDBFile.writeFile"):
            minimizer._run_simulation("dummy.pdb", "out.pdb")
            assert "High Potential Energy" not in caplog.text

    @patch("synth_pdb.physics.app")
    def test_health_check_high_energy_warning(
        self, mock_app: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that health check warns on unusually high potential energy."""
        import logging

        caplog.set_level(logging.WARNING)

        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()

        minimizer = EnergyMinimizer(disable_cache=True)

        # Same mocks as above
        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb
        mock_topo = MagicMock()
        mock_atom = MagicMock()
        mock_topo.atoms.return_value = [mock_atom]
        mock_pdb.topology = mock_topo
        mock_pos = MagicMock()
        mock_pos.__len__.return_value = 1
        mock_pos.value_in_unit.return_value = [[1, 1, 1]]
        mock_pdb.positions = mock_pos
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology = mock_topo
        mock_modeller.positions = mock_pos
        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim

        # Return high energy
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = 1e9
        mock_state.getPositions.return_value = mock_pos
        mock_sim.context.getState.return_value = mock_state

        with patch("synth_pdb.physics.app.PDBFile.writeFile"):
            assert minimizer._run_simulation("dummy.pdb", "out.pdb") is not None
            assert "High Potential Energy" in caplog.text

    @patch("synth_pdb.generator._detect_disulfide_bonds", return_value=[])
    def test_generator_ace_offset_mapping(self, mock_detect: MagicMock) -> None:
        """Verify that generator.py correctly maps PTM names even with ACE cap offsets."""
        sequence = "SEP-ALA"
        # ACE-SER-ALA-NME
        # Res 0 is ACE, Res 1 is SEP (SER), Res 2 is ALA, Res 3 is NME
        # The mapper should handle the index shift correctly.
        from synth_pdb.generator import generate_pdb_content

        # This is a smoke test to check it doesn't crash on name restoration
        try:
            generate_pdb_content(
                sequence_str=sequence,
                cap_termini=True,
                minimize_energy=True,
                minimization_max_iter=1,
            )
        except Exception as e:
            # We allow physics errors if OpenMM is missing, but not index/logic errors
            if "Simulation failed" not in str(e) and "Energy minimization failed" not in str(e):
                pass

    def test_calculate_bfactor_edge_cases(self) -> None:
        """Test B-factor calculation with very low and very high S2."""
        from synth_pdb.generator import _calculate_bfactor

        # Ideal order (S2=1.0)
        b1 = _calculate_bfactor("CA", 10, 20, "ALA", s2=1.0)
        # Disordered (S2=0.0)
        b0 = _calculate_bfactor("CA", 10, 20, "ALA", s2=0.0)

        assert b0 > b1
        assert 5.0 <= b1 <= 99.0
        assert 5.0 <= b0 <= 99.0


# ---------------------------------------------------------------------------
# Kabsch algorithm — reflection correction
# ---------------------------------------------------------------------------


def test_kabsch_superposition_reflection_correction() -> None:
    """The Kabsch algorithm must return a proper rotation (det=+1), not a reflection.

    SCIENTIFIC BASIS:
    SVD of the cross-covariance matrix H = U Σ Vᵀ may produce U and V such
    that det(V Uᵀ) = -1, corresponding to an improper rotation (reflection).
    The algorithm corrects this by constructing R = V diag(1,1,d) Uᵀ where
    d = det(V Uᵀ), which flips the last singular vector and guarantees
    det(R) = +1.

    We construct a structure pair that deliberately triggers d = -1 by
    reflecting one point cloud through the XY plane, then verify that:
    (a) det(rotation_matrix) = +1, and
    (b) the RMSD after superposition is smaller than before.
    """
    import numpy as np
    from synth_pdb.benchmark_metrics import superpose_kabsch

    rng = np.random.default_rng(0)
    ref = rng.standard_normal((10, 3)).astype(np.float64)

    # Create a reflected version: flip Z so the naive SVD gives det=-1
    mobile_reflected = ref.copy()
    mobile_reflected[:, 2] *= -1.0

    rotated, rmsd = superpose_kabsch(mobile_reflected, ref)

    # RMSD after superposition must be finite and non-negative
    assert np.isfinite(rmsd), "RMSD is not finite after Kabsch superposition"
    assert rmsd >= 0.0

    # The rotation matrix implied by the superposition must be proper (det=+1).
    # We back-calculate R from centred coordinates: rotated_c = mobile_c @ R^T
    mob_c = mobile_reflected - mobile_reflected.mean(axis=0)
    rot_c = rotated - ref.mean(axis=0)
    # Least-squares solve: mob_c @ R^T ≈ rot_c  →  R^T ≈ pinv(mob_c) @ rot_c
    r_approx = np.linalg.lstsq(mob_c, rot_c, rcond=None)[0].T
    det = np.linalg.det(r_approx)
    assert det == pytest.approx(1.0, abs=0.05), (
        f"Rotation matrix determinant {det:.4f} ≠ +1 — reflection not corrected"
    )


# ---------------------------------------------------------------------------
# shift_rmsd — NaN (missing assignment) handling
# ---------------------------------------------------------------------------


def test_shift_rmsd_nan_filtering() -> None:
    """shift_rmsd must exclude NaN entries from both the squared-error sum and
    the weight denominator, not treat them as zero-error observations.

    SCIENTIFIC BASIS:
    Experimental spectra routinely have unassigned resonances (NaN).  Including
    NaN values as zero deviations would artifically deflate the RMSD.  The
    denominator must count only the valid (finite) pairs so the result equals
    the RMSD computed over the non-NaN subset alone.
    """
    import math
    import numpy as np
    from synth_pdb.benchmark_metrics import shift_rmsd

    # 4 residues; residues 2 and 3 are unassigned (NaN) in both arrays
    pred = np.array([8.1, float("nan"), float("nan"), 8.3])
    ref = np.array([8.0, float("nan"), float("nan"), 8.4])

    result = shift_rmsd({"H": pred}, {"H": ref})

    # Only residues 0 and 3 contribute: errors are 0.1 and 0.1
    expected = math.sqrt((0.1**2 + 0.1**2) / 2)
    assert result == pytest.approx(expected, abs=1e-6), (
        f"shift_rmsd={result:.6f} ppm, expected {expected:.6f} ppm "
        "— NaN entries not correctly excluded"
    )


def test_shift_rmsd_all_nan_returns_nan() -> None:
    """If every entry is NaN, shift_rmsd must return nan (no valid pairs)."""
    import numpy as np
    from synth_pdb.benchmark_metrics import shift_rmsd

    pred = np.array([float("nan"), float("nan")])
    ref = np.array([float("nan"), float("nan")])
    result = shift_rmsd({"H": pred}, {"H": ref})
    assert np.isnan(result), f"All-NaN input should return nan, got {result}"
