# bmrb_api Module

The `bmrb_api` module provides programmatic access to experimental NMR data from the **Biological Magnetic Resonance Data Bank (BMRB)** and geometric validation metrics from the **Protein Data Bank (PDB)**.

## Overview

A critical part of validating synthetic protein models is comparing them to real-world experimental data. This module allows users to fetch ground-truth restraints, chemical shifts, and geometric quality reports to ensure their synthetic structures are biologically realistic.

### Key Features

-   **Experimental Restraints**: Fetch distance constraints (NOEs) that define the 3D fold of a protein.
-   **Chemical Shifts**: Download peer-reviewed chemical shift assignments for structural validation.
-   **PDBe Validation**: Retrieve summary reports and outlier lists (Ramachandran, bond lengths, etc.) for established PDB entries.
-   **Automated Downloads**: Easily download PDB files from RCSB for benchmarking.

## API Reference

::: synth_pdb.bmrb_api
    handler: python
    options:
      members:
        - BMRBAPI
        - PDBValidationAPI

## Scientific Basis

### BMRB Restraints
In NMR structure calculation, "restraints" are the experimental observations (usually from NOESY experiments) that specify the upper bounds of distances between specific pairs of atoms.
By fetching these for a known protein (like Ubiquitin, BMRB 6457), you can test if `synth-pdb`'s generator or energy minimizer produces structures that satisfy the same constraints as the experimental ensemble.

### Validation Percentiles
The `PDBValidationAPI` provides "percentile scores." A score of 95 means the structure is better than 95% of all structures in the PDB for a given metric (e.g., Ramachandran outliers).

## Usage Example

```python
from synth_pdb.bmrb_api import BMRBAPI, PDBValidationAPI

# 1. Fetch metadata for Human Ubiquitin (BMRB 6457)
metadata = BMRBAPI.get_entry_metadata("6457")
print(f"Title: {metadata.get('title')}")

# 2. Fetch experimental distance restraints
restraints = BMRBAPI.fetch_restraints("6457")
print(f"Found {len(restraints)} distance restraints.")

# 3. Fetch experimental chemical shifts
shifts = BMRBAPI.fetch_chemical_shifts("6457")
if 1 in shifts:
    print(f"Residue 1 NH shift: {shifts[1].get('HN')} ppm")

# 4. Fetch PDBe validation summary for a related PDB (e.g., 1D3Z)
summary = PDBValidationAPI.get_validation_summary("1D3Z")
percentile = summary[0].get("absolute_percentile_clashscore")
print(f"1D3Z Clashscore Percentile: {percentile}")
```
