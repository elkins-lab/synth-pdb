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

    @pytest.fixture(autouse=True)
    def reset_platform_cache(self):
        """Reset the global platform cache before each test to ensure isolation."""
        from synth_pdb.physics import _BEST_PLATFORM_CACHE

        _BEST_PLATFORM_CACHE["platform"] = None
        _BEST_PLATFORM_CACHE["props"] = {}

    @classmethod
    def teardown_class(cls):
        """Final cleanup of the global cache after the entire class finishes."""
        from synth_pdb.physics import _BEST_PLATFORM_CACHE

        _BEST_PLATFORM_CACHE["platform"] = None
        _BEST_PLATFORM_CACHE["props"] = {}

    @pytest.fixture
    def mock_modeller(self):
        """Standard mock for the modeller to bypass PDB processing."""
        modeller = MagicMock()
        modeller.topology = MagicMock()
        modeller.positions = []
        return modeller

    @patch("synth_pdb.physics.mm.Context")
    @patch("synth_pdb.physics.mm.System")
    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_cuda_success_path(
        self, mock_sim, mock_get_platform, mock_sys_cls, mock_ctx_cls, mock_modeller
    ):
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
        # Find the call that used CUDA
        cuda_call = None
        for call in mock_sim.call_args_list:
            if len(call[0]) > 4 and call[0][3] == mock_platform:
                cuda_call = call
                break
        assert cuda_call is not None
        assert cuda_call[0][4] == {"Precision": "mixed"}

    @patch("synth_pdb.physics.mm.Context")
    @patch("synth_pdb.physics.mm.System")
    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_metal_success_path(
        self, mock_sim, mock_get_platform, mock_sys_cls, mock_ctx_cls, mock_modeller
    ):
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
        # Find the call that used Metal
        metal_call = None
        for call in mock_sim.call_args_list:
            if len(call[0]) > 4 and call[0][3] == mock_platform:
                metal_call = call
                break
        assert metal_call is not None
        assert metal_call[0][4] == {"Precision": "single"}

    @patch("synth_pdb.physics.mm.Context")
    @patch("synth_pdb.physics.mm.System")
    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    def test_fail_fast_on_unavailable_platform(
        self, mock_get_platform, mock_sys_cls, mock_ctx_cls, mock_modeller
    ):
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

    @patch("synth_pdb.physics.mm.Context")
    @patch("synth_pdb.physics.mm.System")
    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_auto_detection_fallback_sequence(
        self, mock_sim, mock_get_platform, mock_sys_cls, mock_ctx_cls, mock_modeller
    ):
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
        # Check if any call used OpenCL
        found_opencl = False
        for call in mock_sim.call_args_list:
            if (
                len(call[0]) > 3
                and hasattr(call[0][3], "getName")
                and call[0][3].getName() == "OpenCL"
            ):
                found_opencl = True
                break
        assert found_opencl, "OpenCL platform was not used in any Simulation call"

    @patch("synth_pdb.physics.mm.Context")
    @patch("synth_pdb.physics.mm.System")
    @patch("synth_pdb.physics.mm.Platform.getPlatformByName")
    @patch("synth_pdb.physics.app.Simulation")
    def test_cpu_fallback_when_gpu_init_fails(
        self, mock_sim, mock_get_platform, mock_sys_cls, mock_ctx_cls, mock_modeller
    ):
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
        # We need to find the call that passed exactly 5 arguments (the GPU one)
        gpu_call = mock_sim.call_args_list[0]
        assert gpu_call[0][3].getName() == "CUDA"
        # Second call should be bare (CPU) - only 3 args passed
        cpu_call = mock_sim.call_args_list[1]
        assert len(cpu_call[0]) == 3
