# packing Module

The `packing` module implements a Monte Carlo side-chain optimization engine used to resolve steric clashes and find energetically favorable rotamer conformations.

## Overview

When generating synthetic structures, the backbone is often constructed first. The `packing` module then "packs" the side chains by searching through rotamer libraries—pre-defined collections of common side-chain shapes—to find the configuration that minimizes overlaps (clashes) between atoms.

### Key Features

-   **Monte Carlo Optimization**: Efficiently explores the vast space of possible side-chain combinations.
-   **Rotamer Libraries**: Uses high-quality libraries derived from established protein structures.
-   **Clash Minimization**: Specifically tuned to resolve the "steric overlaps" that occur in raw synthetic models.

## API Reference

::: synth_pdb.packing
    handler: python
    options:
      members:
        - SideChainPacker

## Scientific Principles

### Rotamers and Dihedrals
Side-chain flexibility is mostly limited to rotations around single bonds (dihedral angles $\chi_1, \chi_2, \chi_3, \chi_4$). However, not all angles are equally likely. "Rotamers" are the preferred, low-energy clusters of these angles.

### Monte Carlo Algorithm
To avoid the "combinatorial explosion" of testing every possible side-chain combination, the `SideChainPacker` uses a Monte Carlo approach:
1.  **Initialize**: Randomly assign a rotamer to every residue.
2.  **Perturb**: Select a residue and change its rotamer.
3.  **Evaluate**: Calculate the new clash score.
4.  **Accept/Reject**: If the score improves, always accept. If it gets worse, accept with a probability determined by the "temperature" (Metropolis criterion).

This allows the system to escape local minima and find a globally optimized packing arrangement.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.packing import SideChainPacker

# 1. Generate a structure with "clashing" side chains
gen = PeptideGenerator("MEELQK")
structure = gen.generate(optimize_sidechains=False) # Skip initial packing

# 2. Manually run the packer
packer = SideChainPacker()
optimized_structure = packer.optimize(structure, iterations=500)

print("Side-chain packing complete.")
```
