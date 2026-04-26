# torsion Module

The `torsion` module provides tools for calculating and analyzing the dihedral angles that define protein backbone and side-chain conformations.

## Overview

Dihedral (torsion) angles are the most important degrees of freedom in a protein. This module extracts these angles from `Biotite` structures, enabling Ramachandran analysis and torsion-space modeling.

### Key Features

-   **Backbone Torsions**: Calculation of $\phi$ (phi), $\psi$ (psi), and $\omega$ (omega) angles.
-   **Side-Chain Torsions**: Extraction of $\chi_1, \chi_2, \chi_3, \chi_4$ angles.
-   **Statistics**: Tools for comparing synthetic torsion distributions against experimental libraries.

## API Reference

::: synth_pdb.torsion
    handler: python
    options:
      members:
        - calculate_torsion_angles

## Scientific Principles

### Dihedral Definition
A dihedral angle is defined by four consecutive atoms ($A-B-C-D$). It is the angle between the plane containing $A, B, C$ and the plane containing $B, C, D$.

-   **$\phi$ (Phi)**: $C_{i-1} - N_i - C\alpha_i - C_i$
-   **$\psi$ (Psi)**: $N_i - C\alpha_i - C_i - N_{i+1}$
-   **$\omega$ (Omega)**: $C\alpha_i - C_i - N_{i+1} - C\alpha_{i+1}$

### The Ramachandran Plot
By plotting $\phi$ vs. $\psi$, we can visualize the allowed conformational space of a protein. Helices and sheets appear as distinct clusters on this plot. The `torsion` module provides the raw data needed to generate these visualizations.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.torsion import calculate_torsion_angles

# 1. Generate a structure
gen = PeptideGenerator("MEELQK")
structure = gen.generate(conformation="alpha")

# 2. Calculate backbone torsions
angles = calculate_torsion_angles(structure)

for res in angles:
    print(f"Residue {res['res_id']} ({res['res_name']}):")
    print(f"  Phi: {res['phi']:.1f}°, Psi: {res['psi']:.1f}°")
```
