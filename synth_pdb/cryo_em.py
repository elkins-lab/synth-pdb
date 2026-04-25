"""
Advanced Cryo-EM Density Map Generation for Synthetic Ensembles.
"""

import logging
from typing import Tuple, Union

import biotite.structure as struc
import mrcfile  # type: ignore[import-untyped]
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# EDUCATIONAL OVERVIEW - Cryo-EM Density Simulation:
# -------------------------------------------------
# Cryo-Electron Microscopy (Cryo-EM) measures the Coulomb potential of a
# biological sample. In structural biology, we often represent this as a
# "density map" where each voxel contains a value proportional to the
# expected electron density at that point.
#
# Simulation Principles:
# 1. Atomic Scattering: Each atom (C, N, O, P, S) contributes to the density.
# 2. Gaussian Approximation: At medium resolutions (3-10Å), each atom can be
#    approximated as a Gaussian blob. The width of the Gaussian corresponds to
#    the target resolution of the map.
# 3. Ensemble Averaging: Real Cryo-EM maps are averages of thousands of
#    individual particles. By generating density for a synthetic ensemble,
#    we can simulate "conformational heterogeneity" — regions where the
#    protein is moving will appear blurred or "smeared" in the final map.


def generate_density_map(
    structure: Union[struc.AtomArray, struc.AtomArrayStack],
    resolution: float = 3.0,
    grid_spacing: float = 1.0,
    buffer: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a 3D density map from an AtomArray or AtomArrayStack.

    Args:
        structure: Biotite structure(s) to convert to density.
        resolution: Target resolution in Angstroms (Gaussian sigma ≈ res/3).
        grid_spacing: Voxel size in Angstroms (default 1.0Å).
        buffer: Extra padding around the protein in Angstroms.

    Returns:
        Tuple of (3D density array, (3,) array of grid origin coordinates).

    SCIENTIFIC NOTE - Resolution vs Sigma:
    --------------------------------------
    The relationship between the resolution (R) and the Gaussian sigma (σ)
    is often approximated as σ = R / (2 * sqrt(2 * ln 2)) ≈ R / 2.355.
    However, for simple visual benchmarks, σ = resolution / 3 is also common.
    We use the 1/3 rule here for a conservative "sharpness" at the target res.
    """
    if isinstance(structure, struc.AtomArray):
        # Convert single structure to a stack of 1 for unified processing
        stack = struc.stack([structure])
    else:
        stack = structure

    # 1. Define Grid Boundaries
    # We find the min/max coordinates across ALL models in the ensemble.
    coords = stack.coord
    c_min = np.min(coords, axis=(0, 1)) - buffer
    c_max = np.max(coords, axis=(0, 1)) + buffer

    # Calculate grid dimensions based on spacing
    grid_dims = np.ceil((c_max - c_min) / grid_spacing).astype(int)
    density = np.zeros(grid_dims)

    logger.info(f"Generating Cryo-EM map with dimensions {grid_dims} and spacing {grid_spacing}Å")

    # 2. Voxelization
    # For each atom in each model, we increment the corresponding voxel.
    for model_idx in range(stack.stack_depth()):
        model_coords = stack.coord[model_idx]

        # Calculate voxel indices for all atoms at once
        voxel_indices = ((model_coords - c_min) / grid_spacing).astype(int)

        # Clip indices to ensure they fall within the grid (defensive)
        voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)

        # Accumulate "delta functions" at atomic positions
        # EDUCATIONAL NOTE: In a more advanced implementation, we would
        # weigh atoms by their atomic number (Z) to reflect scattering power.
        for idx in voxel_indices:
            density[idx[0], idx[1], idx[2]] += 1.0

    # 3. Ensemble Averaging
    # We divide by the number of models to get the mean occupancy per voxel.
    density /= stack.stack_depth()

    # 4. Gaussian Blurring (Resolution Simulation)
    # This turns the "point cloud" into a continuous density volume.
    sigma_voxels = (resolution / 3.0) / grid_spacing
    logger.debug(f"Applying Gaussian blur with sigma={sigma_voxels:.2f} voxels")

    blurred_density = gaussian_filter(density, sigma=sigma_voxels)

    return blurred_density, c_min


def save_mrc_file(
    path: str,
    density: np.ndarray,
    origin: np.ndarray,
    spacing: float = 1.0,
) -> None:
    """Saves a 3D density array to an MRC/CCP4 formatted file.

    Args:
        path: Output filename.
        density: 3D numpy array of density values.
        origin: (3,) array of coordinates for the (0,0,0) voxel.
        spacing: Grid spacing in Angstroms.

    EDUCATIONAL NOTE - The MRC Format:
    ---------------------------------
    The MRC format is the standard for 3D electron microscopy data.
    It contains a 1024-byte header followed by the binary density data.
    Key header fields include:
    - NX, NY, NZ: Number of voxels in each dimension.
    - XLEN, YLEN, ZLEN: Physical size of the box in Angstroms.
    - MAPC, MAPR, MAPS: Mapping of array axes to physical axes (usually 1,2,3).
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        # MRC files expect float32
        mrc.set_data(density.astype(np.float32))

        # Set voxel size (Angstroms)
        mrc.voxel_size = spacing

        # Set origin (Standard PDB/MRC translation)
        # Note: mrcfile origin is often stored in the 'origin' header field
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]

        # Ensure the statistics are calculated correctly
        mrc.update_header_stats()

    logger.info(f"Successfully saved Cryo-EM map to {path}")


class CryoEMSimulator:
    """Stateful wrapper for Cryo-EM simulation workflows.

    Allows for reproducible generation of density maps from PDB ensembles.
    """

    def __init__(self, resolution: float = 3.0, spacing: float = 1.0):
        """Initialize the simulator.

        Args:
            resolution: Target map resolution (Å).
            spacing: Voxel spacing (Å).
        """
        self.resolution = resolution
        self.spacing = spacing

    def simulate(self, structure: Union[struc.AtomArray, struc.AtomArrayStack]) -> np.ndarray:
        """Generates a density map for the provided structure."""
        density, _ = generate_density_map(
            structure, resolution=self.resolution, grid_spacing=self.spacing
        )
        return density
