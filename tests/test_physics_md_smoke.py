from unittest.mock import MagicMock, patch

from synth_pdb.physics import simulate_trajectory


def test_simulate_trajectory_smoke_mocked():
    """Smoke test for Molecular Dynamics trajectory simulation using mocks.

    SCIENTIFIC BASIS:
    MD simulations generate a time-series of coordinates (trajectory).
    This test verifies the orchestration of the OpenMM pipeline.
    """
    pdb_content = "HEADER    TEST"  # Minimal content

    # Mock OpenMM components
    with (
        patch("openmm.app.PDBFile") as mock_pdb_file,
        patch("openmm.app.ForceField"),
        patch("openmm.app.Simulation") as mock_sim_cls,
        patch("openmm.LangevinMiddleIntegrator"),
    ):
        # Setup mock simulation instance
        mock_sim = mock_sim_cls.return_value
        mock_context = mock_sim.context

        # Mock state and positions
        mock_state = MagicMock()
        mock_context.getState.return_value = mock_state
        mock_state.getPositions.return_value = []  # Content doesn't matter for mock

        # Mock PDBFile.writeFile to write a fake PDB string to the buffer
        def fake_write_file(topology, positions, file):
            file.write("ATOM      1  N   ALA A   1\nEND")

        mock_pdb_file.writeFile.side_effect = fake_write_file

        # Run simulation: 100 steps, report every 50 -> 3 frames (0, 50, 100)
        trajectory = simulate_trajectory(
            pdb_content=pdb_content, temperature_kelvin=300.0, steps=100, report_interval=50
        )

        assert len(trajectory) == 3
        for frame in trajectory:
            assert "ATOM" in frame
            assert "END" in frame

        # Verify OpenMM calls
        mock_sim.minimizeEnergy.assert_called_once()
        assert mock_sim.step.call_count == 2  # 50 steps twice


def test_simulate_trajectory_empty_on_error():
    """Test that simulate_trajectory returns an empty list on failure."""
    # Pass garbage content to trigger an exception in OpenMM
    with patch("openmm.app.PDBFile", side_effect=Exception("OpenMM Error")):
        trajectory = simulate_trajectory(
            pdb_content="NOT A PDB", temperature_kelvin=300.0, steps=100, report_interval=50
        )
    assert trajectory == []
