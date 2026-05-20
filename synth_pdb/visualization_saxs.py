"""Visualization for SAXS Profiles (Compatibility Shim).

EDUCATIONAL RATIONALE:
======================
Visualization is critical for interpreting Small-Angle X-ray Scattering (SAXS)
data. While the raw 1D scattering curve (I(q) vs q) is the primary output,
several transformed plots are standard in structural biology to emphasize
different aspects of the molecular shape and folding state.

EDUCATIONAL NOTE - The Kratky Plot:
===================================
A Kratky plot (q^2 * I(q) vs q) is used to assess the compactness and folding
state of a protein. For a globular, well-folded protein, the Kratky plot
exhibits a characteristic bell-shaped peak that returns to the baseline.
In contrast, for an intrinsically disordered protein (IDP) or an unfolded
polypeptide, the curve continues to rise or reaches a plateau at high q,
indicating a lack of a compact core.

EDUCATIONAL NOTE - The Guinier Approximation:
=============================================
The Guinier approximation allows for the determination of the radius of
gyration (Rg) from the very low-angle region of the scattering curve.
It states that for small q (specifically q * Rg < 1.3):
    ln(I(q)) ≈ ln(I(0)) - (Rg^2 / 3) * q^2
By plotting ln(I(q)) vs q^2 (a "Guinier Plot"), one can perform a linear
regression where the slope of this line is -Rg^2 / 3. This provides a direct,
model-independent estimate of the overall size of the molecule.

References:
- Guinier, A. (1939). Ann. Phys. 12, 161-237.
- Kratky, O. & Porod, G. (1949). J. Colloid Sci. 4, 35-70.
"""

import logging
from typing import Any

try:
    import synth_saxs as _saxs

    HAS_SYNTH_SAXS = True
except ImportError:
    HAS_SYNTH_SAXS = False

logger = logging.getLogger(__name__)

if HAS_SYNTH_SAXS:
    from synth_saxs import plot_saxs_results
else:

    def plot_saxs_results(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        """Fallback for plot_saxs_results when synth-saxs is missing."""
        raise ImportError("plot_saxs_results requires synth-saxs. Run: pip install synth-saxs")
