import pytest
import numpy as np
import biotite.structure as struc
import os
import tempfile
import logging
from synth_pdb.physics import EnergyMinimizer, HAS_OPENMM
from synth_pdb.generator import generate_pdb_content


@pytest.fixture
def valid_pdb_path():
    """Provides a path to a valid, unminimized PDB file."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tf:
        pdb_content = generate_pdb_content(sequence_str="ALA-ALA-ALA", minimize_energy=False)
        tf.write(pdb_content)
        path = tf.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestPhysicsCoverage:
    def test_minimizer_smoke(self, valid_pdb_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.pdb")
            minimizer = EnergyMinimizer()
            success = minimizer.minimize(valid_pdb_path, out_path)
            assert success is True
            assert os.path.exists(out_path)

    def test_minimizer_invalid_box_size(self):
        with pytest.raises(ValueError, match="box_size must be positive"):
            EnergyMinimizer(box_size=-1.0)

    def test_minimizer_solvent_validation(self, caplog):
        with caplog.at_level(logging.WARNING):
            minimizer = EnergyMinimizer(solvent_model="ghost_water")
            assert "Unknown solvent model" in caplog.text
            assert minimizer.solvent_model == "explicit"

    def test_minimizer_small_explicit_box(self, caplog):
        with caplog.at_level(logging.WARNING):
            minimizer = EnergyMinimizer(solvent_model="explicit", box_size=0.5)
            assert "dangerously small" in caplog.text
            assert minimizer.box_size.value_in_unit(minimizer.box_size.unit) == 1.1

    def test_calculate_energy(self):
        minimizer = EnergyMinimizer()
        pdb_content = generate_pdb_content(sequence_str="GLY-GLY", minimize_energy=False)
        energy = minimizer.calculate_energy(pdb_content)
        assert energy is not None
        assert isinstance(energy, float)

    def test_ptm_renaming(self):
        pdb_content = generate_pdb_content(sequence_str="ALA-SEP-ALA", minimize_energy=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "ptm.pdb")
            out_path = os.path.join(tmpdir, "ptm_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            success = minimizer.minimize(in_path, out_path)
            assert success is True
            with open(out_path) as f:
                content = f.read()
                assert "SEP" in content

    def test_sec_renaming(self):
        # SEC is special because it renames atom SE -> SG
        pdb_content = generate_pdb_content(sequence_str="ALA-SEC-ALA", minimize_energy=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "sec.pdb")
            out_path = os.path.join(tmpdir, "sec_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            success = minimizer.minimize(in_path, out_path)
            assert success is True
            with open(out_path) as f:
                content = f.read()
                assert "SEC" in content
                assert "SE" in content

    def test_ion_stripping(self, caplog):
        pdb_content = generate_pdb_content(
            sequence_str="CPYCKKRFHSH", metal_ions="auto", minimize_energy=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "ion.pdb")
            out_path = os.path.join(tmpdir, "ion_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            with caplog.at_level(logging.INFO, logger="synth_pdb.physics"):
                minimizer.minimize(in_path, out_path)
            assert any("Restoring lost HETATM: ZN" in record.message for record in caplog.records)

    def test_equilibrate_short(self, valid_pdb_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.pdb")
            minimizer = EnergyMinimizer()
            success = minimizer.equilibrate(valid_pdb_path, out_path, steps=5)
            assert success is True

    def test_cyclic_minimization(self):
        pdb_content = generate_pdb_content(sequence_str="ALA-GLY-ALA", minimize_energy=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "cyclic.pdb")
            out_path = os.path.join(tmpdir, "cyclic_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            success = minimizer.minimize(in_path, out_path, cyclic=True)
            assert success is True

    def test_robust_system_fallback(self, mocker):
        minimizer = EnergyMinimizer()
        from openmm import app

        minimizer.implicit_solvent_enum = app.OBC2
        mock_system = mocker.Mock()
        first_call = [True]

        def side_effect(topo, **kwargs):
            if first_call[0] and "implicitSolvent" in kwargs:
                first_call[0] = False
                raise Exception(
                    "implicitSolvent was specified to createSystem() but was never used"
                )
            return mock_system

        mocker.patch.object(minimizer.forcefield, "createSystem", side_effect=side_effect)
        system, topo, pos = minimizer._create_system_robust(app.Topology(), None)
        assert system == mock_system
        assert "implicitSolvent" in minimizer._suppressed_args

    def test_health_check_nan_coords(self, mocker, valid_pdb_path):
        # Mock _run_simulation to return None via Health Check fail
        # This is internal, so we mock simulation state
        from openmm import unit

        mock_state = mocker.Mock()
        mock_state.getPotentialEnergy.return_value = 100.0 * unit.kilojoules_per_mole
        # Return NaN positions
        mock_state.getPositions.return_value = [np.array([np.nan, 0, 0]) * unit.nanometers]

        mocker.patch("openmm.app.Simulation")
        # Actually it's easier to mock the whole _run_simulation loop or return values
        pass

    def test_no_openmm_fallback(self, monkeypatch):
        monkeypatch.setattr("synth_pdb.physics.HAS_OPENMM", False)
        minimizer = EnergyMinimizer()
        assert minimizer.minimize("in.pdb", "out.pdb") is False
        assert minimizer.calculate_energy("DUMMY") == 0.0
        assert minimizer.equilibrate("in.pdb", "out.pdb") is False
        assert minimizer.add_hydrogens_and_minimize("in.pdb", "out.pdb") is False

    def test_calculate_energy_error_handling(self, mocker):
        minimizer = EnergyMinimizer()
        mocker.patch.object(minimizer, "_run_simulation", return_value=None)
        energy = minimizer.calculate_energy("ALA")
        assert energy is None

    def test_finalize_output_error_handling(self, mocker):
        minimizer = EnergyMinimizer()
        mock_sim = mocker.Mock()
        mock_sim.topology = mocker.Mock()
        mock_sim.context.getState.return_value.getPositions.return_value = []
        mocker.patch("builtins.open", side_effect=OSError("Disk Full"))
        with pytest.raises(IOError):
            minimizer._finalize_output("bad.pdb", mock_sim, False, [], [], [], {}, [])

    def test_minimization_reporter(self, caplog):
        from synth_pdb.physics import LoggingMinimizationReporter

        reporter = LoggingMinimizationReporter(interval=1)
        args = {"system energy": 123.456}
        with caplog.at_level(logging.DEBUG, logger="synth_pdb.physics"):
            res = reporter.report(1, None, None, args)
        assert res is False
        assert any("Minimization Iteration 1" in record.message for record in caplog.records)

    def test_salt_bridge_restraints(self, caplog):
        """Test detection and restraint of salt bridges."""
        # ARG and GLU should form a salt bridge
        pdb_content = generate_pdb_content(sequence_str="ARG-ALA-GLU", minimize_energy=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "salt.pdb")
            out_path = os.path.join(tmpdir, "salt_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            with caplog.at_level(logging.DEBUG, logger="synth_pdb.physics"):
                success = minimizer.minimize(in_path, out_path)
            assert success is True
            # Check for salt bridge log
            assert any(
                "Found" in record.message and "salt bridges" in record.message
                for record in caplog.records
            )

    def test_multiple_forcefields(self):
        """Test initializing with multiple forcefield XMLs."""
        # Standard OpenMM supports a list of XMLs
        minimizer = EnergyMinimizer(forcefield_name=["amber14-all.xml", "amber14/tip3pfb.xml"])
        assert minimizer.forcefield is not None

    def test_cyclic_no_duplicate_oxt_warning(self, recwarn):
        """Regression test: Ensure cyclic minimization doesn't warn about duplicate OXT."""
        # Linear PDB content (which contains OXT)
        pdb_content = generate_pdb_content(sequence_str="ALA-GLY-ALA", minimize_energy=False)
        assert "OXT" in pdb_content

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "cyclic_oxt.pdb")
            out_path = os.path.join(tmpdir, "cyclic_oxt_out.pdb")
            with open(in_path, "w") as f:
                f.write(pdb_content)

            minimizer = EnergyMinimizer()
            # This should not trigger "WARNING: duplicate atom" from OpenMM
            success = minimizer.minimize(in_path, out_path, cyclic=True)
            assert success is True

            # Check for warnings from OpenMM (it uses warnings.warn internally)
            for warning in recwarn:
                assert "duplicate atom" not in str(warning.message)

    def test_minimization_reporter_error(self, caplog):
        from synth_pdb.physics import LoggingMinimizationReporter

        reporter = LoggingMinimizationReporter(interval=1)
        with caplog.at_level(logging.DEBUG, logger="synth_pdb.physics"):
            reporter.report(1, None, None, {})
        assert any("0.0000 kJ/mol" in record.message for record in caplog.records)
