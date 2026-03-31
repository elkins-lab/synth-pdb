"""Chemical Shift prediction for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import predict_chemical_shifts, calculate_csi

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr.chemical_shifts as _cs

RANDOM_COIL_SHIFTS = _cs.RANDOM_COIL_SHIFTS
SECONDARY_SHIFTS = _cs.SECONDARY_SHIFTS
# Private functions used in tests
_calculate_ring_current_shift = _cs._calculate_ring_current_shift
_get_aromatic_rings = _cs._get_aromatic_rings
calculate_csi = _cs.calculate_csi
get_secondary_structure = _cs.get_secondary_structure
predict_chemical_shifts = _cs.predict_chemical_shifts

__all__ = [
    "predict_chemical_shifts",
    "calculate_csi",
    "get_secondary_structure",
    "RANDOM_COIL_SHIFTS",
    "SECONDARY_SHIFTS",
    "_calculate_ring_current_shift",
    "_get_aromatic_rings",
]
