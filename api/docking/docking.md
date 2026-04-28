# docking Module

The `docking` module provides utilities for preparing synthetic protein structures for molecular docking simulations and electrostatic calculations.

## Overview

Molecular docking requires structures to be in specific formats that include physical parameters beyond simple coordinates. The `docking` module focuses on converting standard PDB files into **PQR format**, which includes partial atomic charges and Van der Waals radii.

### Key Features

-   **PQR Export**: Converts PDB files to PQR format using OpenMM for charge assignment.
-   **Charge Assignment**: Automatically adds missing hydrogens and assigns partial charges based on the `AMBER` forcefield.
-   **Radii Calculation**: Derives atomic radii from Lennard-Jones parameters ($\sigma / 2$).
-   **PTM Handling**: Standardizes non-standard or modified residues for compatibility with standard forcefields.

## API Reference

::: synth_pdb.docking
    handler: python
    options:
      members:
        - DockingPrep

## Scientific Principles

### The PQR Format
The PQR format is a variation of PDB where:
1.  The **Occupancy** column is replaced by the **Partial Charge** ($q$) of the atom.
2.  The **B-factor** column is replaced by the **Atomic Radius** ($r$) in Angstroms.

This format is the standard input for electrostatic solvers like **APBS** (Adaptive Poisson-Boltzmann Solver) and many docking engines.

### Charge Assignment Logic
The module uses `OpenMM` to:
1.  **Standardize**: Map modified residues (like phosphorylated Serine) to their parent residues if necessary.
2.  **Protonate**: Add hydrogens at a specific pH (default 7.4).
3.  **Parameterize**: Apply a forcefield (default `amber14-all.xml`) to look up the partial charge for each atom type in its specific chemical environment.

## Usage Example

```python
from synth_pdb.docking import DockingPrep

# 1. Initialize the preparation tool
prep = DockingPrep(forcefield_name="amber14-all.xml")

# 2. Convert a synthetic PDB to PQR
# This will add hydrogens and assign charges
success = prep.write_pqr(
    input_pdb="synthetic_protein.pdb", 
    output_pqr="ready_for_docking.pqr"
)

if success:
    print("PQR file created successfully.")
```
