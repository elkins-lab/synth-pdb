"""NMR Relaxation calculations for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_relaxation_rates, predict_order_parameters

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE - NMR Relaxation and the Lipari-Szabo Model
============================================================
NMR relaxation rate measurements (T1, T2, NOE) provide information about the
timescales and amplitudes of protein motion. These motions range from fast
bond vibrations (picoseconds) to slow loop rearrangements (microseconds).

THE MODEL-FREE FORMALISM:
Lipari and Szabo (1982) introduced a "model-free" approach to describe
protein dynamics using only two key parameters:
1. S^2 (Order Parameter): Measures the spatial amplitude of internal motion.
   S^2=1 means rigid, S^2=0 means fully isotropic.
2. taue (Effective Correlation Time): Measures the speed of the internal motion.

APPLICATION:
For a globular protein, the overall tumbling (taum) usually dominates the
relaxation rate. By measuring relaxation at multiple magnetic field strengths,
researchers can de-convolve the global and local motions. synth-pdb uses
S^2 parameters predicted from sequence and structure to generate synthetic
T1/T2 rates that reflect the protein's dynamic landscape.

"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr as _nmr
import synth_nmr.relaxation as _rel

calculate_relaxation_rates = _nmr.calculate_relaxation_rates
predict_order_parameters = _nmr.predict_order_parameters
njit = _rel.njit
spectral_density = _rel.spectral_density

__all__ = [
    "calculate_relaxation_rates",
    "predict_order_parameters",
    "spectral_density",
    "njit",
]
