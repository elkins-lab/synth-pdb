"""
Geometry utilities for protein structure generation and analysis.
"""

from synth_pdb.geometry._numba import njit
from synth_pdb.geometry.dihedral import (
    calculate_angle,
    calculate_dihedral,
    calculate_dihedral_angle,
)
from synth_pdb.geometry.nerf import (
    place_atom,
    position_atom_3d_from_internal_coords,
)
from synth_pdb.geometry.rmsd import (
    calculate_average_coords,
    calculate_pairwise_rmsd,
    calculate_rmsd,
    calculate_rmsd_to_average,
)
from synth_pdb.geometry.sidechain import (
    reconstruct_sidechain,
    rotate_points,
)
from synth_pdb.geometry.superposition import (
    apply_transformation,
    find_medoid,
    kabsch_superposition,
    superimpose_structures,
)
from synth_pdb.geometry.vectorized import (
    batched_angle,
    batched_dihedral,
    position_atoms_batch,
    superimpose_batch,
)

__all__ = [
    "calculate_angle",
    "calculate_dihedral",
    "calculate_dihedral_angle",
    "position_atom_3d_from_internal_coords",
    "place_atom",
    "calculate_average_coords",
    "calculate_pairwise_rmsd",
    "calculate_rmsd",
    "calculate_rmsd_to_average",
    "reconstruct_sidechain",
    "rotate_points",
    "apply_transformation",
    "find_medoid",
    "kabsch_superposition",
    "superimpose_structures",
    "batched_angle",
    "batched_dihedral",
    "position_atoms_batch",
    "superimpose_batch",
    "njit",
]
