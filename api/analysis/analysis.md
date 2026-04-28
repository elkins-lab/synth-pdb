# analysis Module

The `analysis` module provides high-level tools for evaluating structural quality, comparing protein models, and analyzing conformational ensembles.

## Overview

After generating synthetic structures or ensembles, it is often necessary to quantify their precision and accuracy. This module provides a suite of geometric analyzers that handle alignment, RMSD calculation, and residue-level strain analysis.

### Key Features

-   **Structure Comparison**: Optimal superposition of PDB models using the Kabsch algorithm.
-   **Ensemble Analysis**: Identification of the "medoid" structure (most representative) and calculation of ensemble-averaged RMSD.
-   **Geometric Strain**: Identification of localized structural distortions, such as non-trans peptide bonds.

## API Reference

::: synth_pdb.analysis
    handler: python
    options:
      members:
        - GeometryAnalyzer

## Scientific Principles

### Kabsch Superposition
To compare two structures, we must find the rotation matrix $R$ and translation vector $\mathbf{t}$ that minimize the Root Mean Square Deviation (RMSD):

$$RMSD = \sqrt{\frac{1}{N} \sum_{i=1}^N \| R\mathbf{x}_i + \mathbf{t} - \mathbf{y}_i \|^2}$$

The `GeometryAnalyzer` uses the Kabsch algorithm (via SVD) to find the optimal $R$ that aligns the mobile structure $\mathbf{x}$ to the reference structure $\mathbf{y}$.

### Ensemble Medoid
For an ensemble of NMR structures or MD frames, the **medoid** is the structure $k$ that has the minimum average RMSD to all other structures in the set:

$$\text{Medoid} = \arg\min_k \frac{1}{N} \sum_{j=1}^N RMSD(k, j)$$

This is a more robust representative of the ensemble than a simple average structure, which may have physically impossible bond lengths or angles.

## Usage Example

```python
from synth_pdb.analysis import GeometryAnalyzer

# 1. Compare two structures (e.g., predicted vs. ground truth)
results = GeometryAnalyzer.compare_pdbs(
    "model.pdb", 
    "reference.pdb", 
    ca_only=True
)
print(f"RMSD: {results['rmsd']:.2f} Å")

# 2. Analyze an NMR ensemble
ensemble_files = ["frame1.pdb", "frame2.pdb", "frame3.pdb"]
stats = GeometryAnalyzer.analyze_ensemble_pdbs(ensemble_files)
print(f"Ensemble Precision: {stats['avg_rmsd']:.2f} Å")
print(f"Most representative model: {stats['medoid_path']}")

# 3. Check for geometric strain
strain = GeometryAnalyzer.calculate_residue_strain("model.pdb")
for res_id, dev in strain.items():
    if dev > 20: # Large deviation from trans
        print(f"Warning: High omega strain at residue {res_id}: {dev:.1f}°")
```
