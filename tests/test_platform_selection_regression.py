import os
import pytest
import openmm as mm
from synth_pdb.physics import EnergyMinimizer, _BEST_PLATFORM_CACHE
from synth_pdb.generator import generate_pdb_content


def test_platform_selection_gpu_to_cpu_fallback_regression(mocker):
    """
    Regression test for platform selection logic (0fae34d).
    Ensures that when GPU platforms (CUDA, OpenCL, Metal) are unavailable,
    the system falls back to OpenMM's default (platform=None) instead of
    explicitly pinning 'Reference' or 'CPU'.
    """
    # 1. Clear the cache to ensure auto-detection runs
    _BEST_PLATFORM_CACHE["platform"] = None
    _BEST_PLATFORM_CACHE["props"] = {}

    # 2. Mock mm.Platform.getPlatformByName to fail for GPUs
    # We want to simulate a system where CUDA/Metal/OpenCL are either
    # not installed or broken.
    def mock_get_platform(name):
        if name in ["CUDA", "OpenCL", "Metal"]:
            raise Exception(f"Platform {name} not available")
        return mm.Platform.getPlatformByName(name)

    mocker.patch("openmm.Platform.getPlatformByName", side_effect=mock_get_platform)

    # 3. Mock GITHUB_ACTIONS=true to simulate CI environment
    mocker.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"})

    # 4. Trigger platform detection via EnergyMinimizer
    # We need a small PDB to run a dummy simulation
    minimizer = EnergyMinimizer()

    # Mocking _run_simulation to actually check what platform it ends up with
    # or just let it run if it's fast enough.
    # Actually, we want to check the CACHE after a detection attempt.
    # EnergyMinimizer._run_simulation is where the detection logic lives.

    # We'll use a real file and run a minimal minimization to trigger the logic.
    import tempfile

    with (
        tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp_in,
        tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp_out,
    ):
        tmp_in.write(generate_pdb_content(length=3, minimize_energy=False))
        tmp_in.flush()

        # This will trigger the detection logic in _run_simulation
        minimizer.minimize(tmp_in.name, tmp_out.name, max_iterations=1)

    # 5. Assertions
    # The cache should STILL have platform=None because all probes failed.
    # This ensures we don't accidentally cache "Reference" or "CPU".
    assert _BEST_PLATFORM_CACHE["platform"] is None

    # Also verify that we didn't force "Reference" into the cache
    # (Checking the getName if it were not None)
    if _BEST_PLATFORM_CACHE["platform"] is not None:
        assert _BEST_PLATFORM_CACHE["platform"].getName() not in ["Reference", "CPU"]


def test_platform_selection_explicit_request(mocker):
    """
    Ensures that if a user explicitly requests 'Reference', they get it,
    but it doesn't get CACHED as the 'best' platform for subsequent runs.
    """
    _BEST_PLATFORM_CACHE["platform"] = None

    minimizer = EnergyMinimizer(platform_name="Reference")

    import tempfile

    with (
        tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp_in,
        tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp_out,
    ):
        tmp_in.write(generate_pdb_content(length=3, minimize_energy=False))
        tmp_in.flush()

        minimizer.minimize(tmp_in.name, tmp_out.name, max_iterations=1)

    # Cache should remain None because it was an explicit request, not auto-detection
    assert _BEST_PLATFORM_CACHE["platform"] is None
