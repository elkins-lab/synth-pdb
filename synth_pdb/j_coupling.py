"""J-Coupling calculations for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_hn_ha_coupling

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — Scalar J-Couplings and Karplus Equations
============================================================
Scalar J-couplings (indirect dipole-dipole interactions) are mediated through
chemical bonds. In proteins, the three-bond 3J(HN-HA) coupling is the most
widely used observable for determining backbone φ (phi) torsion angles.

THE KARPLUS EQUATION:
The relationship between the J-coupling and the torsion angle (θ) is
described by the empirical Karplus equation:

    3J(θ) = A · cos²θ + B · cosθ + C

where A, B, and C are constants derived from fitting experimental values
of proteins with known crystal structures (e.g., A=6.51, B=-1.76, C=1.60
from Hu & Bax, 1997). For 3J(HN-HA), the torsion angle θ is |φ - 60°|.

APPLICATIONS:
1. Secondary Structure Refinement: Helices (φ ≈ -60°) give small J-couplings
   (< 6 Hz), while β-sheets (φ ≈ -120°) give large J-couplings (> 8 Hz).
2. Side-chain Rotamers: 3J(N-HB) and 3J(C'-HB) define the χ1 angle.
3. Dynamical Averaging: Large deviation from these ideal values often
   indicates structural heterogeneity or rapid conformational exchange.

"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr as _nmr

calculate_hn_ha_coupling = _nmr.calculate_hn_ha_coupling

__all__ = ["calculate_hn_ha_coupling"]
