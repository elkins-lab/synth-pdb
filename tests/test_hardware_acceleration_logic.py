import pytest
from unittest.mock import MagicMock, patch
import openmm as mm
from synth_pdb.physics import EnergyMinimizer


class TestHardwareAccelerationLogic:
    """
    Tier 1 Tests: Verifying hardware acceleration logic via mocking.
    These tests simulate different hardware environments (CUDA, Metal, etc.)
    to ensure code paths are followed correctly without needing physical GPUs.
    """

    @pytest.fixture
    def mock_modeller(self):
        """Standard mock for the modeller to bypass PDB processing."""
        modeller = MagicMock()
        modeller.topology = MagicMock()
        modeller.positions = []
        return modeller

    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_cuda_success_path(self, mock_sim, mock_get_platform, mock_modeller):
        """Simulate a machine with functional CUDA."""
        # 1. Setup Mock
        mock_platform = MagicMock(spec=mm.Platform)
        mock_platform.getName.return_value = "CUDA"
        mock_get_platform.return_value = mock_platform

        # 2. Act: Explicitly request CUDA
        minimizer = EnergyMinimizer(platform_name="CUDA", precision="mixed", disable_cache=True)

        # Mock ALL internal system creation to avoid building a real system
        with patch.object(minimizer, "_create_system_robust") as mock_sys:
            mock_sys.return_value = (MagicMock(), MagicMock(), [])
            with patch.object(minimizer.forcefield, "createSystem") as mock_fsys:
                mock_fsys.return_value = MagicMock()
                minimizer._build_simulation_context(mock_modeller, False, [], [], [], [])

        # 3. Assert: Verify CUDA was used with 'mixed' precision
        mock_get_platform.assert_called_with("CUDA")
        called_props = mock_sim.call_args[0][4]
        assert called_props == {"Precision": "mixed"}

    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_metal_success_path(self, mock_sim, mock_get_platform, mock_modeller):
        """Simulate a machine with functional Metal (Apple Silicon)."""
        mock_platform = MagicMock(spec=mm.Platform)
        mock_platform.getName.return_value = "Metal"
        mock_get_platform.return_value = mock_platform

        minimizer = EnergyMinimizer(platform_name="Metal", precision="single", disable_cache=True)

        with patch.object(minimizer, "_create_system_robust") as mock_sys:
            mock_sys.return_value = (MagicMock(), MagicMock(), [])
            with patch.object(minimizer.forcefield, "createSystem") as mock_fsys:
                mock_fsys.return_value = MagicMock()
                minimizer._build_simulation_context(mock_modeller, False, [], [], [], [])

        mock_get_platform.assert_called_with("Metal")
        called_props = mock_sim.call_args[0][4]
        assert called_props == {"Precision": "single"}

    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    def test_fail_fast_on_unavailable_platform(self, mock_get_platform, mock_modeller):
        """Verify that explicitly requesting an unavailable platform raises RuntimeError."""
        # Simulate platform not found
        mock_get_platform.side_effect = Exception("Platform not found")

        minimizer = EnergyMinimizer(platform_name="CUDA", disable_cache=True)

        # Mock system creation to avoid crashing before the platform check
        with patch.object(minimizer, "_create_system_robust") as mock_sys:
            mock_sys.return_value = (MagicMock(), MagicMock(), [])
            with patch.object(minimizer.forcefield, "createSystem") as mock_fsys:
                mock_fsys.return_value = MagicMock()
                with pytest.raises(
                    RuntimeError, match="Requested platform 'CUDA' is not available"
                ):
                    minimizer._build_simulation_context(mock_modeller, False, [], [], [], [])

    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_auto_detection_fallback_sequence(self, mock_sim, mock_get_platform, mock_modeller):
        """Simulate a machine where CUDA is broken, but OpenCL works."""

        def side_effect(name):
            if name == "CUDA":
                raise Exception("Broken drivers")
            if name == "Metal":
                raise Exception("Not a Mac")
            if name == "OpenCL":
                p = MagicMock()
                p.getName.return_value = "OpenCL"
                return p
            raise Exception("None left")

        mock_get_platform.side_effect = side_effect

        # No explicit platform requested
        minimizer = EnergyMinimizer(disable_cache=True)

        with patch.object(minimizer, "_create_system_robust") as mock_sys:
            mock_sys.return_value = (MagicMock(), MagicMock(), [])
            with patch.object(minimizer.forcefield, "createSystem") as mock_fsys:
                mock_fsys.return_value = MagicMock()
                minimizer._build_simulation_context(mock_modeller, False, [], [], [], [])

        # Should have tried CUDA, then Metal, then settled on OpenCL
        assert mock_get_platform.call_count >= 3
        actual_platform = mock_sim.call_args[0][3]
        assert actual_platform.getName() == "OpenCL"

    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_cpu_fallback_when_gpu_init_fails(self, mock_sim, mock_get_platform, mock_modeller):
        """Verify that if a GPU is detected but fails to INITIALIZE (e.g. bad Context), we fallback to CPU."""
        mock_platform = MagicMock()
        mock_platform.getName.return_value = "CUDA"
        mock_get_platform.return_value = mock_platform

        # First call (with GPU) fails, second call (CPU fallback) succeeds
        mock_sim.side_effect = [Exception("GPU Context Failed"), MagicMock()]

        minimizer = EnergyMinimizer(disable_cache=True)  # Auto-detect mode

        with patch.object(minimizer, "_create_system_robust") as mock_sys:
            mock_sys.return_value = (MagicMock(), MagicMock(), [])
            with patch.object(minimizer.forcefield, "createSystem") as mock_fsys:
                mock_fsys.return_value = MagicMock()
                minimizer._build_simulation_context(mock_modeller, False, [], [], [], [])

        # First call used CUDA
        assert mock_sim.call_args_list[0][0][3].getName() == "CUDA"
        # Second call should be bare (CPU) - no 4th arg passed
        assert len(mock_sim.call_args_list[1][0]) == 3
