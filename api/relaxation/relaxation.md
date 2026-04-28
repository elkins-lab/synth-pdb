# relaxation Module

The `relaxation` module provides utilities for calculating NMR relaxation rates and order parameters.

## Overview

NMR relaxation is a primary source of information about protein dynamics. This module provides tools to calculate these rates from synthetic structures, leveraging the [synth-nmr](https://github.com/elkins/synth-nmr) engine.

## Main Functions

::: synth_pdb.relaxation.calculate_relaxation_rates
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.relaxation.predict_order_parameters
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.relaxation.spectral_density
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Predicting Order Parameters

```python
from synth_pdb.relaxation import predict_order_parameters

# structure: biotite.structure.AtomArray
s2 = predict_order_parameters(structure)

# s2: List[float] (per-residue S2 values)
```

### Calculating Relaxation Rates

Calculate R1, R2, and NOE from structural coordinates and magnetic field strength.

```python
from synth_pdb.relaxation import calculate_relaxation_rates

rates = calculate_relaxation_rates(
    structure,
    field_mhz=600.0,
    correlation_time_ns=10.0
)

# rates: Dict[str, np.ndarray] (r1, r2, noe)
```

## References

- **Lipari-Szabo Model-Free**: Lipari, G., & Szabo, A. (1982). "Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules." *Journal of the American Chemical Society*. [DOI: 10.1021/ja00381a009](https://doi.org/10.1021/ja00381a009)
- **NMR Dynamics**: Palmer, A. G. (2004). "NMR characterization of the dynamics of macromolecules." *Chemical Reviews*. [DOI: 10.1021/cr030413t](https://doi.org/10.1021/cr030413t)

## See Also

- [nmr Module](nmr.md) - General NMR utilities
- [chemical_shifts Module](chemical_shifts.md) - NMR chemical shift prediction
- [Scientific Background: NMR Theory](../science/nmr-theory.md)
- [Scientific Background: IDP Dynamics](../science/idp-dynamics.md)
