# validator Module

The `validator` module provides comprehensive geometric and biophysical validation for protein structures.

## Overview

The `validator` module implements standard checks to ensure generated structures adhere to physical and biological principles. It is often used to "pre-screen" structures before expensive physics-based refinement.

## Main Classes

::: synth_pdb.validator.PDBValidator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - validate_all
        - validate_bond_lengths
        - validate_bond_angles
        - validate_ramachandran
        - validate_steric_clashes
        - validate_peptide_plane
        - validate_sequence_improbabilities
        - validate_side_chain_rotamers
        - validate_chirality

## Validation Checks

### Bond Geometry

Standard bond lengths (N-CA, CA-C, C-N, C-O) and angles are checked against equilibrium values with configurable tolerances.

### Ramachandran Analysis

Dihedral angles ($\phi, \psi$) are validated against **MolProbity-style** polygonal regions. The module distinguishes between:
- **Favored**: 98% of high-quality protein structures.
- **Allowed**: 99.8% of high-quality protein structures.
- **Outlier**: Highly improbable conformations.

Specific polygons are used for **Glycine**, **Proline**, and **Pre-Proline** residues.

### Steric Clashes

Detects overlapping atoms that are too close in space.
- **Minimum Distance**: Atoms must be ≥ 0.5 Å apart.
- **VdW Overlap**: Atoms are checked against their Van der Waals radii.
- **CA-CA Spacing**: C-alpha atoms of non-consecutive residues must be ≥ 3.0 Å apart.

### Side-Chain Rotamers

Validates side-chain conformations against the **Dunbrack Backbone-Dependent Rotamer Library**. Rotamers that deviate significantly from favored staggered conformations are flagged.

## Usage Examples

### Basic Validation

```python
from synth_pdb.validator import PDBValidator

# Load PDB content
with open("peptide.pdb") as f:
    pdb_content = f.read()

# Run validator
validator = PDBValidator(pdb_content)
validator.validate_all()

# Get violations
violations = validator.get_violations()
for v in violations:
    print(v)
```

## Educational Notes

### Physics of the Ramachandran Plot

The Ramachandran plot is the "map of allowed protein shapes". It is primarily based on **hard-sphere sterics**:

1. **The Clash**: For most $\phi/\psi$ angles, the carbonyl oxygen of residue $i$ clashes with the amide hydrogen of residue $i+1$, or the side-chain $C_\beta$ atom clashes with the backbone.
2. **The Exceptions**:
    - **Glycine**: Has no $C_\beta$ atom, allowing it to access regions that are "illegal" for other amino acids. It serves as a flexible hinge.
    - **Proline**: Its cyclic side-chain locks its $\phi$ angle to $\sim -65^\circ$, acting as a structural stiffener.

### Side-Chain Packing (Rotamers)

Side chains are not free to rotate continuously. They snap into specific discrete conformations called **Rotamers** (Rotational Isomers) to minimize steric repulsion. These are typically staggered conformations ($gauche^+$, $gauche^-$, $trans$).

## References

- **Ramachandran Plot**: Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963). "Stereochemistry of polypeptide chain configurations." *Journal of Molecular Biology*. [DOI: 10.1016/S0022-2836(63)80023-6](https://doi.org/10.1016/S0022-2836(63)80023-6)
- **MolProbity (Top8000)**: Williams, C. J., et al. (2018). "MolProbity: More and better reference data for improved all-atom structure validation." *Protein Science*. [DOI: 10.1002/pro.3330](https://doi.org/10.1002/pro.3330)
- **Rotamer Library**: Dunbrack, R. L., & Cohen, F. E. (1997). "Bayesian statistical analysis of protein side-chain rotamer preferences." *Protein Science*. [DOI: 10.1002/pro.5560060802](https://doi.org/10.1002/pro.5560060802)
- **Calpha Geometry**: Lovell, S. C., et al. (2003). "Structure validation by Calpha geometry: phi,psi and Cbeta deviation." *Proteins*. [DOI: 10.1002/prot.10286](https://doi.org/10.1002/prot.10286)

## See Also

- [physics Module](physics.md) - Physics-based refinement
- [biophysics Module](biophysics.md) - Chemical property refinement
- [Scientific Background: Ramachandran Plots](../science/ramachandran.md)
- [Scientific Background: Rotamer Libraries](../science/rotamers.md)
