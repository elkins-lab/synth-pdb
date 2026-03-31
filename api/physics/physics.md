# physics Module

The `physics` module provides high-performance physics-based refinement for synthetic protein structures using the [OpenMM](https://openmm.org/) engine.

## Overview

While the [generator](generator.md) creates structures based on geometric rules, the `physics` module ensures these structures are physically plausible by resolving steric clashes and optimizing bond lengths and angles through energy minimization.

## Main Classes

::: synth_pdb.physics.EnergyMinimizer
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - minimize
        - equilibrate
        - add_hydrogens_and_minimize
        - calculate_energy

## Usage Examples

### Energy Minimization

Refine a PDB structure to resolve steric clashes and regularize geometry.

```python
from synth_pdb.physics import EnergyMinimizer

minimizer = EnergyMinimizer(solvent_model="obc2")
success = minimizer.minimize("input.pdb", "output_minimized.pdb")
```

### Thermal Equilibration (MD)

Run a short Molecular Dynamics (MD) simulation to "heat" the protein to 300K, allowing it to settle into a stable dynamic average.

```python
minimizer.equilibrate("input.pdb", "output_equilibrated.pdb", steps=1000)
```

### Handling Metal Coordination

Automatically apply harmonic constraints to coordinate metal ions like Zinc (Zn2+).

```python
# Metal-ion coordination logic is handled automatically when calling minimize
# with the appropriate parameters derived from the generator or cofactors module.
minimizer.minimize("input.pdb", "output.pdb", coordination=[("ZN", [10, 15, 20, 25])])
```

## Educational Notes

### What is Energy Minimization?

Proteins fold into specific 3D shapes to minimize their "Gibbs Free Energy". A generated structure often has "clashes" where atoms are too close (high Van der Waals repulsion) or bond angles are strained.

Energy Minimization is like rolling a ball down a hill. The "Energy Landscape" represents the potential energy of the protein as a function of all its atom coordinates. The algorithm moves atoms slightly to reduce this energy, finding a local minimum where the structure is physically relaxed.

### Anatomy of a Forcefield

A forcefield (like **Amber14**) approximates the potential energy ($U$) of a molecule as a sum of four main terms:

$$U = U_{bond} + U_{angle} + U_{torsion} + [U_{vdw} + U_{elec}]$$

1. **Bonded Terms**: Atoms behave like balls on springs. Pushing them away from ideal lengths/angles costs energy.
2. **Non-Bonded Terms**:
    - **Van der Waals (Lennard-Jones)**: Models steric repulsion and London dispersion.
    - **Electrostatics (Coulomb)**: Interaction between point charges (e.g., salt bridges).

### Implicit vs. Explicit Solvent

1. **Explicit Solvent (TIP3P)**: Every water molecule ($H_2O$) is simulated. This captures the full enthalpic and entropic costs of solvation but is computationally expensive.
2. **Implicit Solvent (OBC2)**: Also known as "Born Solvation". The water is treated as a continuous dielectric field. The **OBC2 (Onufriev-Bashford-Case)** model is a refined version that parameterizes atomic radii to match explicit solvent behavior closely while being 10-100x faster.

## References

- **OpenMM**: Eastman, P., et al. (2017). "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics." *PLOS Computational Biology*. [DOI: 10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)
- **OBC2 Solvent**: Onufriev, A., Bashford, D., & Case, D. A. (2004). "Exploring protein native states and large-scale conformational changes with a modified generalized born model." *Proteins: Structure, Function, and Bioinformatics*. [DOI: 10.1002/prot.20154](https://doi.org/10.1002/prot.20154)
- **Amber Forcefield**: Case, D. A., et al. (2005). "The Amber biomolecular simulation programs." *Journal of Computational Chemistry*. [DOI: 10.1002/jcc.20290](https://doi.org/10.1002/jcc.20290)
- **Lipari-Szabo Formalism**: Lipari, G., & Szabo, A. (1982). "Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules." *Journal of the American Chemical Society*. [DOI: 10.1021/ja00381a009](https://doi.org/10.1021/ja00381a009)

## See Also

- [validator Module](validator.md) - Pre-minimization geometric validation
- [biophysics Module](biophysics.md) - pH and salt bridge logic
- [Scientific Background: Energy Minimization](../science/energy-minimization.md)
