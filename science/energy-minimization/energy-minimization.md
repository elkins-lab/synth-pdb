# Energy Minimization

This document explains the physical principles and implementation of energy minimization in `synth-pdb`.

## What is Energy Minimization?

Proteins fold into specific 3D shapes to minimize their **Gibbs Free Energy**. A generated structure, especially one built from simple geometry or random sampling, often contains "clashes" (where atoms are too close) or strained bond angles and lengths.

Energy minimization is a computational process used to "relax" these structures. It treats the protein as a collection of atoms and uses a **Force Field** to calculate the potential energy of the system as a function of atomic coordinates. The algorithm then iteratively moves the atoms to find a local minimum on the energy landscape.

## Implementation with OpenMM

`synth-pdb` leverages [OpenMM](https://openmm.org/), a high-performance toolkit for molecular simulation, to perform energy minimization.

### Force Fields

A force field is the set of parameters and mathematical functions used to calculate the potential energy. `synth-pdb` defaults to the **AMBER14** force field (`amber14-all.xml`), which provides high-quality parameters for proteins, including:
-   **Bond terms:** Harmonic potentials for bond lengths and angles.
-   **Torsion terms:** Periodic functions for dihedral angles.
-   **Non-bonded terms:** Van der Waals (Lennard-Jones) and electrostatic (Coulomb) interactions.

### Solvent Models

The environment surrounding the protein significantly affects its energy. `synth-pdb` supports several solvent models:

1.  **Implicit Solvent (Generalized Born / OBC):**
    -   The effect of water is modeled as a continuous medium with a high dielectric constant (ε ≈ 80).
    -   The **OBC2 (Onufriev-Bashford-Case)** model is the default, offering a good balance between speed and accuracy.
    -   Ideal for rapid refinement and NMR-style structure regularizations.

2.  **Explicit Solvent (TIP3P):**
    -   Individual water molecules are explicitly included in a simulation box.
    -   Captures detailed hydrogen bonding and entropic effects of solvation.
    -   Requires more computational resources but provides the highest fidelity.

## Usage in `synth-pdb`

To run energy minimization on a generated structure, use the `--minimize` flag:

```bash
python -m synth_pdb.main --sequence "MEELQK" --minimize --solvent obc2
```

### Advanced Refinement

-   **`--refine-clashes`**: A lightweight, purely geometric alternative that adjusts clashing atoms without a full physics engine. Useful when OpenMM is not available.
-   **`--equilibrate`**: Runs a short Molecular Dynamics (MD) simulation after minimization to allow the structure to sample the local conformational space at a specific temperature (default 300K).
-   **`--cyclic`**: Automatically applies constraints and minimization to ensure the N- and C-termini of a cyclic peptide meet in 3D space.

## NMR Perspective

In NMR structure calculation (e.g., using CYANA or XPLOR-NIH), energy minimization is a critical final step. It ensures that the structures satisfying experimental restraints (like NOEs and J-couplings) also have excellent covalent geometry and no steric overlaps, satisfying the requirements for "high-quality" structural models.
