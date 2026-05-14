"""J-Coupling calculations for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_hn_ha_coupling

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE - Scalar J-Couplings and Karplus Equations
============================================================
Scalar J-couplings (indirect dipole-dipole interactions) are mediated through
chemical bonds. In proteins, the three-bond 3J(HN-HA) coupling is the most
widely used observable for determining backbone phi (phi) torsion angles.

THE KARPLUS EQUATION:
The relationship between the J-coupling and the torsion angle (theta) is
described by the empirical Karplus equation:

    3J(theta) = A * cos^2theta + B * costheta + C

where A, B, and C are constants derived from fitting experimental values
of proteins with known crystal structures (e.g., A=6.51, B=-1.76, C=1.60
from Hu & Bax, 1997). For 3J(HN-HA), the torsion angle theta is |phi - 60deg|.

APPLICATIONS:
1. Secondary Structure Refinement: Helices (phi ~ -60deg) give small J-couplings
   (< 6 Hz), while beta-sheets (phi ~ -120deg) give large J-couplings (> 8 Hz).
2. Side-chain Rotamers: 3J(N-HB) and 3J(C'-HB) define the chi1 angle.
3. Dynamical Averaging: Large deviation from these ideal values often
   indicates structural heterogeneity or rapid conformational exchange.

"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr as _nmr

calculate_hn_ha_coupling = _nmr.calculate_hn_ha_coupling

__all__ = ["calculate_hn_ha_coupling"]
