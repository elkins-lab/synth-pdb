"""NMR Spectroscopy utilities for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_synthetic_noes

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — The Nuclear Overhauser Effect (NOE)
======================================================
The NOE is the transfer of magnetization between two nuclear spins that are
close to each other in space (< 6 Å), regardless of whether they are bonded.
Because the intensity of the NOE is proportional to the inverse sixth power
of the distance (1/r⁶), it provides highly sensitive distance constraints
for structural determination.

TYPES OF NOE DATA:
1. Long-range NOEs: Between residues far apart in sequence but close in space.
   These define the tertiary structure (fold).
2. Short-range NOEs: Between residues close in sequence (e.g., i to i+1).
   These define secondary structure (e.g., helices and sheets).

NOE DISTANCE RESTRAINTS:
In structure calculation, NOEs are typically converted into distance upper
bounds. For example, a "strong" NOE cross-peak usually implies a distance of
less than 2.5 Å, a "medium" peak < 3.5 Å, and a "weak" peak < 5.0 Å.
synth-pdb simulates these interactions precisely using the structure's
coordinates and provides the distances needed for validation.

"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr as _nmr

calculate_synthetic_noes = _nmr.calculate_synthetic_noes

__all__ = ["calculate_synthetic_noes"]
