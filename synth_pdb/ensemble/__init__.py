"""
NMR ensemble analysis algorithms.

Provides scientifically general tools for analysing structural ensembles
extracted from pdbstat-python and validated against published NMR benchmarks.

Submodules:
    daop       - Dihedral Angle Order Parameters (Hyberts et al. 1992)
    statistics - Typed dataclasses for ensemble statistics and quality assessment

Examples:
    >>> import numpy as np
    >>> from synth_pdb.ensemble import DAOPCalculator, EnsembleStatistics

    >>> # Order-parameter calculation
    >>> angles = np.full(20, -np.pi / 3)  # perfectly ordered phi angles
    >>> S = DAOPCalculator.calculate_order_parameter(angles)
    >>> assert S == 1.0

    >>> # Typed statistics container
    >>> stats = EnsembleStatistics(n_models=20, n_residues=91,
    ...                            mean_pairwise_rmsd=0.72, rmsd_to_mean=0.68,
    ...                            pct_well_defined=88.0)
    >>> print(stats.precision)   # 'HIGH'
"""

from synth_pdb.ensemble.daop import DAOPCalculator
from synth_pdb.ensemble.statistics import EnsembleStatistics, QualityAssessment

__all__ = [
    "DAOPCalculator",
    "EnsembleStatistics",
    "QualityAssessment",
]
