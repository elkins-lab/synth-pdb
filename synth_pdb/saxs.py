"""Small-Angle X-ray Scattering (SAXS) for synth-pdb.

This module provides a compatibility shim that re-exports the SAXS simulation
engine from the synth-saxs package.

For direct usage of SAXS functionality, consider using synth-saxs directly:
    pip install synth-saxs
    from synth_saxs import calculate_saxs_profile

See: https://github.com/elkins/synth-saxs

EDUCATIONAL OVERVIEW - SAXS Curve Simulation:
==============================================================
Small-Angle X-ray Scattering (SAXS) is a fundamental technique for studying
protein structure and dynamics in solution. It measures the scattering of
X-rays by the electrons in the sample at small angles (typically 0.1 to 10 degrees).

The resulting 1D curve (Intensity vs. Scattering Vector q) contains information
about the global size, shape, and folding state of the molecule.

SCIENTIFIC PRINCIPLES:
----------------------
1. The Debye Formula: Relates the 3D atomic coordinates to the 1D scattering
   intensity via interference between all atom pairs.
2. Solvent Contrast: Proteins are measured in solution, so the scattering
   from the displaced solvent volume must be subtracted.
3. Atomic Form Factors: Element-specific scattering efficiencies (q-dependent).

References:
- Waasmaier, D. & Kirfel, A. (1995). Acta Cryst. A51, 416-431.
- Pavlov, M.Y. & Svergun, D.I. (1997). J. Appl. Cryst. 30, 712-717.
"""

import logging
from typing import Any

try:
    import synth_saxs as _saxs

    HAS_SYNTH_SAXS = True
except ImportError:
    HAS_SYNTH_SAXS = False

logger = logging.getLogger(__name__)

# Re-exports for backward compatibility
if HAS_SYNTH_SAXS:
    from synth_saxs import (
        SaxsSimulator,
        calculate_radius_of_gyration,
        calculate_saxs_profile,
        export_saxs_profile,
        get_form_factor,
    )
else:
    logger.warning(
        "synth-saxs package not found. SAXS functionality will be unavailable. "
        "Install it via: pip install synth-saxs"
    )

    # Fallback placeholders to prevent import errors but log warnings on use
    def calculate_saxs_profile(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError("calculate_saxs_profile requires synth-saxs. Run: pip install synth-saxs")

    def calculate_radius_of_gyration(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        from biotite.structure import gyration_radius

        return gyration_radius(*args, **kwargs)

    class SaxsSimulator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("SaxsSimulator requires synth-saxs. Run: pip install synth-saxs")

    def export_saxs_profile(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError("export_saxs_profile requires synth-saxs. Run: pip install synth-saxs")

    def get_form_factor(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError("get_form_factor requires synth-saxs. Run: pip install synth-saxs")
