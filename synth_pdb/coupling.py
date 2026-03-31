"""Coupling utilities for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_hn_ha_coupling, predict_couplings_from_structure

The J-coupling (scalar coupling) is a fundamental NMR parameter that provides
information about the chemical bonding and local geometry of a molecule.
In proteins, the 3J(HN-HA) coupling is particularly useful as it relates to the
phi torsion angle via the Karplus equation.

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr.j_coupling as _j

calculate_hn_ha_coupling = _j.calculate_hn_ha_coupling_from_phi
predict_couplings_from_structure = _j.predict_couplings_from_phi_map

__all__ = [
    "calculate_hn_ha_coupling",
    "predict_couplings_from_structure",
]
