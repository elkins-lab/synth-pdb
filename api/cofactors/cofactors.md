# cofactors Module

The `cofactors` module handles the detection and modeling of inorganic cofactors and metal ions within protein structures.

## Overview

Many proteins require metal ions (like Zn²⁺, Ca²⁺, or Mg²⁺) for structural stability or catalytic function. This module automatically identifies coordination motifs—such as Zinc Fingers—and inserts the appropriate ions with correct geometric coordination.

### Key Features

-   **Motif Detection**: Scans the structure for residues capable of coordinating metals (Cys, His, Asp, Glu).
-   **Geometric Coordination**: Places metal ions at the geometric centroid of their coordinating ligands.
-   **Automated Insertion**: Integrates with the generation pipeline to produce holoproteins (proteins with cofactors).

## API Reference

::: synth_pdb.cofactors
    handler: python
    options:
      members:
        - find_metal_binding_sites
        - add_metal_ion

## Scientific Principles

### Coordination Chemistry
In biological systems, metal ions are coordinated by specific amino acid side chains (ligands). For example, a **Zinc Finger** typically coordinates a Zn²⁺ ion using two Cysteines and two Histidines ($C_2H_2$).

The module calculates the ideal position for the metal ion as the centroid ($\mathbf{C}$) of the coordinating atoms:

$$\mathbf{C} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i$$

where $\mathbf{x}_i$ are the coordinates of the ligand atoms (e.g., SG for Cys, NE2/ND1 for His).

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.cofactors import find_metal_binding_sites, add_metal_ion

# 1. Generate a sequence known to bind Zinc (e.g., a simple Zinc Finger)
seq = "CPYCGKSFSDSSALRNHVRTH"
gen = PeptideGenerator(seq)
structure = gen.generate(conformation="alpha")

# 2. Detect potential binding sites
sites = find_metal_binding_sites(structure)

# 3. Add the metal ion if a site is found
if sites:
    structure_with_metal = add_metal_ion(structure, sites[0])
```
