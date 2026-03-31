# chemical_shifts Module

The `chemical_shifts` module provides utilities for predicting NMR chemical shifts from protein structures.

## Overview

Chemical shifts are highly sensitive indicators of local secondary structure and environment. This module provides shims to [synth-nmr](https://github.com/elkins/synth-nmr) for predicting these values.

## Main Functions

::: synth_pdb.chemical_shifts.predict_chemical_shifts
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.chemical_shifts.calculate_csi
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.chemical_shifts.get_secondary_structure
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Predicting Chemical Shifts

```python
from synth_pdb.chemical_shifts import predict_chemical_shifts

# structure: biotite.structure.AtomArray
shifts = predict_chemical_shifts(structure)

# shifts: Dict[int, Dict[str, float]]
# {residue_id: {atom_name: shift_value}}
```

### Chemical Shift Index (CSI)

Analyze secondary structure based on H-alpha, C-alpha, and C-beta shifts.

```python
from synth_pdb.chemical_shifts import calculate_csi

csi = calculate_csi(shifts)
# Returns list of -1 (alpha), 1 (beta), 0 (coil)
```

## References

- **Random Coil Shifts**: Wishart, D. S., et al. (1995). "1H, 13C and 15N random coil NMR chemical shifts of the common amino acids. I. Investigations of nearest-neighbor effects." *Journal of Biomolecular NMR*. [DOI: 10.1007/BF00211783](https://doi.org/10.1007/BF00211783)
- **Ring Current Effects**: Haigh, C. W., & Mallion, R. B. (1979). "Ring current theories in nuclear magnetic resonance." *Progress in Nuclear Magnetic Resonance Spectroscopy*. [DOI: 10.1016/0079-6565(79)80010-2](https://doi.org/10.1016/0079-6565(79)80010-2)

## See Also

- [nmr Module](nmr.md) - General NMR utilities
- [relaxation Module](relaxation.md) - NMR dynamics and relaxation
- [Scientific Background: NMR Theory](../science/nmr-theory.md)
