"""Structure utilities for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import get_secondary_structure

See: https://github.com/elkins/synth-nmr

EDUCATIONAL NOTE — NMR Structure Utilities and RMSD
====================================================
Structural determination via NMR often involves generating an ensemble of
possible conformations (e.g., 20 models) rather than a single crystal-like
structure. Several metrics are used to evaluate and compare these ensembles.

ROOT-MEAN-SQUARE DEVIATION (RMSD):
The RMSD measures the average distance between the atoms (usually CA) of two
superimposed structures. For an NMR ensemble, the "ensemble RMSD" is the
average pairwise RMSD between all models, or the RMSD of each model to the
mean coordinate set.

    RMSD = √[ (1/N) * Σ (r_i,1 - r_i,2)² ]

where N is the number of atoms, and r_i,1 and r_i,2 are the coordinates of
the corresponding atom in the two structures after optimal superposition.

THE MEDOID MODEL:
In an NMR ensemble, the "medoid" is the single model that is most similar to
all others (i.e., it has the lowest average RMSD to all other models). This
is often preferred over the "mean structure" (average coordinates) because
the medoid is a physically realistic structure with correct bond lengths
and angles, whereas the mean structure may have distorted geometry.

synth-pdb's structure utilities provide fast, vectorized engines for these
calculations, ensuring that generated ensembles meet the rigorous standards
of NMR structural biology.

"""

# Re-export from synth-nmr for backward compatibility
import synth_nmr.structure_utils as _su

get_secondary_structure = _su.get_secondary_structure

__all__ = ["get_secondary_structure"]
