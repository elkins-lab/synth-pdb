# Rotamer Libraries

This document explains the use of rotamer libraries for side-chain placement in `synth-pdb`.

## What is a Rotamer?

A rotamer (short for rotational isomer) is a preferred conformation of a protein side chain. Side chains are relatively flexible, but steric constraints mean they often prefer specific, low-energy dihedral angles (χ1, χ2, etc.).

## Side-Chain Dihedral Angles (Chi)

Side-chain flexibility is defined by the rotation about single bonds, denoted as Chi (χ) angles:
-   **χ1:** N–Cα–Cβ–Cγ
-   **χ2:** Cα–Cβ–Cγ–Cδ
-   -   and so on, depending on the length of the side chain.

## Implementation in `synth-pdb`

`synth-pdb` uses two types of rotamer libraries:

1.  **Backbone-Independent Rotamer Library:** A collection of the most frequent side-chain conformations observed in the Protein Data Bank (PDB), regardless of the backbone's φ and ψ angles.
2.  **Backbone-Dependent Rotamer Library:** A more advanced library that provides the probability of each rotamer based on the specific backbone dihedral angles (φ, ψ). This library is more accurate as the backbone conformation can significantly influence the allowed side-chain orientations.

## Placement Process

The tool places side-chains in several steps:

1.  **Initial Placement:** For each residue, a rotamer is sampled from the library (either random or the most probable).
2.  **Optimization:** If the `--optimize` flag is used, the tool performs a Monte Carlo search through the rotamer library to find a combination of side-chain conformations that minimizes steric clashes across the entire protein structure.

## Scientific Relevance

Accurate side-chain placement is crucial for:
-   **Clash avoidance:** Preventing atoms from overlapping in 3D space.
-   **Predicting NMR observables:** Chemical shifts and NOE restraints are highly sensitive to the local environment and distance between side-chain atoms.
-   **Hydrogen bonding:** Many side chains act as hydrogen bond donors or acceptors.

For researchers in NMR spectroscopy, the exact orientation of side chains (especially aromatic rings and methyl groups) is essential for interpreting NOE cross-peaks and predicting relaxation parameters.
