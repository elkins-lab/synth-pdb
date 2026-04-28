# special_chemistry Module

The `special_chemistry` module models complex post-translational modifications (PTMs) and unique chemical events that go beyond standard amino acid chains.

## Overview

Some proteins undergo autocatalytic chemical changes that are essential for their function. A prime example is the Green Fluorescent Protein (GFP), where a specific tripeptide sequence (`SYG`, `TYG`, or `GYG`) undergoes cyclization and oxidation to form a fluorophore.

### Key Features

-   **GFP Chromophore Maturation**: Models the cyclization of the `Ser65-Tyr66-Gly67` motif.
-   **Covalent Bond Manipulation**: Tools for programmatically adding or removing bonds between atoms in a `Biotite` structure.

## API Reference

::: synth_pdb.special_chemistry
    handler: python
    options:
      members:
        - find_gfp_chromophore_motif
        - form_gfp_chromophore

## Scientific Principles

### GFP Chromophore Formation
The maturation of the GFP chromophore involves three steps:
1.  **Cyclization**: The amide nitrogen of Gly67 attacks the carbonyl carbon of Ser65.
2.  **Dehydration**: Loss of a water molecule to form a five-membered heterocyclic ring (imidazolin-5-one).
3.  **Oxidation**: Dehydrogenation of the Tyr66 $C\alpha-C\beta$ bond to create a conjugated system.

The `special_chemistry` module simulates the final structural state of this matured chromophore, allowing for realistic modeling of fluorescent proteins.

## Usage Example

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.special_chemistry import (
    find_gfp_chromophore_motif, 
    form_gfp_chromophore
)

# 1. Generate a sequence containing the GFP motif
# (A fragment of the GFP barrel)
seq = "FEGUFSYGVQCFS" 
gen = PeptideGenerator(seq)
structure = gen.generate(conformation="alpha")

# 2. Identify the chromophore motif (SYG)
motif = find_gfp_chromophore_motif(structure)

# 3. Apply the chemical modification
if motif:
    matured_structure = form_gfp_chromophore(structure, motif)
```
