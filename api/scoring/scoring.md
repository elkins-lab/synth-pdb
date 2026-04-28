# scoring Module

The `scoring` module provides biophysical scoring functions for evaluating the quality and plausibility of synthetic protein models.

## Overview

Unlike the `physics` module, which calculates continuous forces for minimization, the `scoring` module provides discrete metrics that describe a structure's "health." These scores are used to select the best models from an ensemble or to validate a final structure.

### Key Features

-   **Clash Score**: Quantifies the severity of steric overlaps between atoms.
-   **Packing Density**: Measures how well-packed the protein interior is.
-   **Solvent Accessibility**: Identifies residues on the surface vs. the core.

## API Reference

::: synth_pdb.scoring
    handler: python
    options:
      members:
        - calculate_clash_score

## Scientific Principles

### Steric Clashes
Two atoms are said to "clash" if the distance between them is significantly less than the sum of their Van der Waals radii. The `calculate_clash_score` function identifies these overlaps:

$$Score = \sum_{i,j} \max(0, (R_i + R_j) - d_{ij} - \text{threshold})$$

A score of 0.0 indicates a physically plausible structure with no significant steric violations.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.scoring import calculate_clash_score

# 1. Generate a raw structure
gen = PeptideGenerator("MEELQK")
structure = gen.generate(optimize_sidechains=False)

# 2. Evaluate the clash score
score = calculate_clash_score(structure)
print(f"Initial Clash Score: {score:.2f}")

if score > 10.0:
    print("Warning: Structure has significant steric clashes.")
```
