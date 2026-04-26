# saxs Module

The `saxs` module enables the simulation of Small-Angle X-ray Scattering (SAXS) profiles from atomic structures and ensembles.

## Overview

SAXS is a technique used to probe the global shape, size, and flexibility of proteins in solution. This module uses the **Debye Formula** to calculate the scattering intensity $I(q)$ as a function of the scattering vector magnitude $q$.

### Key Features

-   **Debye Formula Implementation**: Accurate $O(N^2)$ calculation of interference patterns.
-   **Atomic Form Factors**: Uses $q$-dependent Gaussian approximations for C, N, O, S, P, and H.
-   **Solvent Contrast**: Accounts for the scattering of the displaced solvent volume (hydration shell approximation).
-   **Ensemble Averaging**: Computes the mean scattering profile for flexible structures or structural ensembles.

## API Reference

::: synth_pdb.saxs
    handler: python
    options:
      members:
        - calculate_saxs_profile
        - export_saxs_profile
        - SaxsSimulator

## Scientific Principles

### The Debye Formula
The total scattering intensity $I(q)$ is computed by summing the interference between all pairs of atoms $i$ and $j$:

$$I(q) = \sum_i \sum_j f_i(q) f_j(q) \frac{\sin(q r_{ij})}{q r_{ij}}$$

where:
-   $q$ is the scattering vector ($q = \frac{4\pi \sin \theta}{\lambda}$).
-   $r_{ij}$ is the distance between atoms $i$ and $j$.
-   $f_i(q)$ is the effective atomic form factor.

### Solvent Subtraction
In solution, we measure the "excess" scattering of the protein. The module approximates the effective form factor as:

$$f_{eff}(q) = f_{vac}(q) - \rho_{sol} V_i \exp\left(-\frac{q^2 V_i^{2/3}}{4\pi}\right)$$

where $\rho_{sol}$ is the electron density of the solvent and $V_i$ is the atomic volume.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.saxs import calculate_saxs_profile, export_saxs_profile

# 1. Generate a structure
gen = PeptideGenerator("NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQ")
structure = gen.generate(conformation="alpha")

# 2. Simulate SAXS profile
q, intensity = calculate_saxs_profile(
    structure, 
    q_min=0.0, 
    q_max=0.5, 
    n_points=100
)

# 3. Export to .dat file
export_saxs_profile(q, intensity, "protein_saxs.dat")
```
