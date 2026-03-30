# geometry

The `geometry` module provides a comprehensive suite of utilities for 3D coordinate manipulation, structural comparison, and protein reconstruction.

## Dihedral and Angle
Calculations for bond angles and torsion angles.

- `calculate_angle(coord1, coord2, coord3)`: Calculates the angle (in degrees) formed by three coordinates.
- `calculate_dihedral(p1, p2, p3, p4)`: Calculates the dihedral angle (in degrees) defined by four points.

## RMSD
Root Mean Square Deviation calculations for structural comparison.

- `calculate_rmsd(P, Q)`: Calculate RMSD between two sets of coordinates.
- `calculate_pairwise_rmsd(coords_list, superimpose=False)`: Calculate pairwise RMSD matrix for multiple structures.
- `calculate_average_coords(coords_list)`: Calculate average (centroid) coordinates from an ensemble.
- `calculate_rmsd_to_average(coords_list)`: Calculate RMSD of each structure to the average structure.

## Superposition
Optimal alignment of structures using the Kabsch algorithm.

- `kabsch_superposition(P, Q)`: Calculate optimal rotation and translation to superimpose P onto Q.
- `apply_transformation(coords, rotation, translation)`: Apply rotation and translation to coordinates.
- `superimpose_structures(mobile, reference)`: Superimpose mobile structure onto reference structure.
- `find_medoid(coords_list, superimpose=True)`: Find the most representative structure (medoid) from an ensemble.

## NeRF Geometry
Natural Extension Reference Frame (NeRF) for building 3D structures from internal coordinates.

- `position_atom_3d_from_internal_coords(p1, p2, p3, bond_length, bond_angle, dihedral)`: Place a new atom in 3D space.

## Sidechain Reconstruction
Tools for rebuilding and rotating sidechain conformations.

- `reconstruct_sidechain(peptide, res_id, rotamer)`: Updates sidechain coordinates in place using rotamer angles.

## Vectorized Operations
High-performance batch processing for large ensembles.

- `superimpose_batch(sources, targets)`: Batched Kabsch algorithm.
- `position_atoms_batch(p1, p2, p3, lengths, angles, dihedrals)`: Batched NeRF algorithm.
- `batched_angle(p1, p2, p3)`: Vectorized angle calculations.
- `batched_dihedral(p1, p2, p3, p4)`: Vectorized dihedral calculations.

For scientific details and mathematical background, see the [NeRF Geometry Background](../science/nerf-geometry.md) page.
