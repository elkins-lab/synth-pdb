# cryo_em Module

The `cryo_em` module provides tools for generating synthetic 3D density maps from protein structures and ensembles. This is essential for simulating how conformational heterogeneity and resolution limits affect experimental Cryo-EM data.

## Overview

Cryo-Electron Microscopy (Cryo-EM) measures the Coulomb potential of a biological sample. In `synth-pdb`, we simulate this by representing each atom as a Gaussian blob whose width corresponds to the target resolution.

### Key Features

-   **Gaussian Voxelization**: Converts atomic coordinates into a 3D grid of density values.
-   **Ensemble Averaging**: Supports `AtomArrayStack` to simulate how flexible regions appear blurred in a consensus map.
-   **MRC Export**: Saves density maps in the industry-standard MRC/CCP4 format for visualization in ChimeraX or PyMOL.

## API Reference

::: synth_pdb.cryo_em
    handler: python
    options:
      members:
        - generate_density_map
        - save_mrc_file
        - CryoEMSimulator

## Scientific Principles

### Resolution vs. Sigma
The relationship between the reported resolution ($R$) and the Gaussian standard deviation ($\sigma$) used for blurring is:

$$\sigma = \frac{R}{3}$$

This conservative "1/3 rule" ensures that the density reflects the structural details appropriate for the target resolution. At 3Å resolution, atoms are clearly separated; at 8Å, only secondary structure elements (like alpha helices) remain visible as "tubes" of density.

### Conformational Heterogeneity
When simulating an ensemble (IDPs or flexible loops), the density at each voxel is the average occupancy across all models:

$$\rho(\mathbf{r}) = \frac{1}{N} \sum_{i=1}^N \rho_i(\mathbf{r})$$

This naturally results in lower, more diffuse density for highly mobile regions, mimicking the "B-factor" or local resolution effects seen in real experiments.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.cryo_em import generate_density_map, save_mrc_file

# 1. Generate an ensemble of decoys
gen = PeptideGenerator("MEELQK")
ensemble = gen.generate_ensemble(n_models=50)

# 2. Generate a 4Å density map
density, origin = generate_density_map(
    ensemble, 
    resolution=4.0, 
    grid_spacing=1.0
)

# 3. Save to MRC for visualization
save_mrc_file("synthetic_map.mrc", density, origin, spacing=1.0)
```
